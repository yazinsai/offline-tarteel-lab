/**
 * NDJSON line protocol: stdin = one audio path per line, stdout = one JSON result per line.
 * Keeps ONNX + trie loaded across many tier-2 samples in one Python process.
 */

import * as readline from "node:readline";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { beamSearchDecode } from "../src/worker/beam-decode.ts";
import { buildTrie, type CompactTrie } from "../src/lib/phoneme-trie.ts";
import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import type { TranscribeResult, BeamVerseMatch } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function makeTranscribe(
  decoder: CTCDecoder,
  trie: CompactTrie | null,
): (audio: Float32Array) => Promise<TranscribeResult> {
  return async (audio: Float32Array) => {
    const { features, timeFrames } = await computeMelSpectrogram(audio);
    const numMels = 80;
    const { logprobs, timeSteps, vocabSize } = await runInference(features, numMels, timeFrames);
    const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
    let beamMatches: BeamVerseMatch[] | undefined;
    if (trie) {
      const beamResults = beamSearchDecode(
        logprobs,
        timeSteps,
        vocabSize,
        decoder.getBlankId(),
        trie,
        8,
      );
      const seen = new Set<string>();
      beamMatches = [];
      for (const result of beamResults) {
        for (const ref of result.matchedVerses) {
          const key = `${ref.verseIndex}:${ref.spanLength}`;
          if (!seen.has(key)) {
            seen.add(key);
            beamMatches.push({
              verseIndex: ref.verseIndex,
              spanLength: ref.spanLength,
              score: result.score,
            });
          }
        }
      }
    }
    return {
      ...greedy,
      acoustic: {
        logprobs,
        timeSteps,
        vocabSize,
        blankId: decoder.getBlankId(),
      },
      beamMatches,
    };
  };
}

async function firstVerseMatch(audioPath: string, db: QuranDB, transcribe: (a: Float32Array) => Promise<TranscribeResult>): Promise<{ surah: number; ayah: number }> {
  const audio = loadAudio(audioPath);
  const tracker = new RecitationTracker(db, transcribe);
  const messages: WorkerOutbound[] = [];
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
    const chunk = audio.slice(offset, end);
    messages.push(...(await tracker.feed(chunk)));
  }
  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);
  for (let i = 0; i < silenceChunks; i++) {
    messages.push(...(await tracker.feed(silenceChunk)));
  }
  for (const msg of messages) {
    if (msg.type === "verse_match") {
      return { surah: msg.surah, ayah: msg.ayah };
    }
  }
  return { surah: 0, ayah: 0 };
}

async function main() {
  const modelPath = resolve(ROOT, "public/fastconformer_phoneme_q8.onnx");
  await createSession(modelPath);
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  const decoder = new CTCDecoder(vocabJson);
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  const built = buildTrie(quranData, vocabJson, 3);
  const trie = built.trie;
  const transcribe = makeTranscribe(decoder, trie);

  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    const p = line.trim();
    if (!p) continue;
    try {
      const { surah, ayah } = await firstVerseMatch(p, db, transcribe);
      process.stdout.write(JSON.stringify({ surah, ayah }) + "\n");
    } catch (e) {
      process.stdout.write(JSON.stringify({ error: String(e), surah: 0, ayah: 0 }) + "\n");
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
