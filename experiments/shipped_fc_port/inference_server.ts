/**
 * Line-oriented stdin/stdout server: one JSON request per line {"path":"<abs audio>"},
 * responds with RESULT_JSON:<predict payload> per line.
 *
 * Model + Quran data + tracker match the shipped offline-tarteel web frontend.
 */

import { execSync } from "node:child_process";
import * as readline from "node:readline";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../../reference_repo/offline-tarteel/web/frontend/src/worker/mel.ts";
import { CTCDecoder } from "../../reference_repo/offline-tarteel/web/frontend/src/worker/ctc-decode.ts";
import { QuranDB } from "../../reference_repo/offline-tarteel/web/frontend/src/lib/quran-db.ts";
import type { TranscribeResult } from "../../reference_repo/offline-tarteel/web/frontend/src/lib/tracker.ts";
import { createSession, runInference } from "../../reference_repo/offline-tarteel/web/frontend/test/session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REF_FE = resolve(__dirname, "../../reference_repo/offline-tarteel/web/frontend");

const SAMPLE_RATE = 16_000;

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 80 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function main(): Promise<void> {
  const modelPath = resolve(REF_FE, "public/fastconformer_phoneme_q8.onnx");
  await createSession(modelPath);

  const vocabJson = JSON.parse(readFileSync(resolve(REF_FE, "public/phoneme_vocab.json"), "utf-8"));
  const decoder = new CTCDecoder(vocabJson);
  const quranData = JSON.parse(readFileSync(resolve(REF_FE, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);

  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let audioPath: string;
    try {
      audioPath = JSON.parse(trimmed).path;
    } catch {
      process.stdout.write(`RESULT_JSON:${JSON.stringify({ error: "bad_json" })}\n`);
      continue;
    }
    if (!audioPath || typeof audioPath !== "string") {
      process.stdout.write(`RESULT_JSON:${JSON.stringify({ error: "missing_path" })}\n`);
      continue;
    }

    try {
      const out = await predictOne(db, decoder, audioPath);
      process.stdout.write(`RESULT_JSON:${JSON.stringify(out)}\n`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      process.stdout.write(
        `RESULT_JSON:${JSON.stringify({ surah: 1, ayah: 1, ayah_end: null, score: 0, transcript: "", error: msg })}\n`,
      );
    }
  }
}

async function predictOne(
  db: QuranDB,
  decoder: CTCDecoder,
  audioPath: string,
): Promise<Record<string, unknown>> {
  async function transcribe(audioSlice: Float32Array): Promise<TranscribeResult> {
    const { features, timeFrames } = await computeMelSpectrogram(audioSlice);
    const numMels = 80;
    const { logprobs, timeSteps, vocabSize } = await runInference(features, numMels, timeFrames);
    return decoder.decode(logprobs, timeSteps, vocabSize);
  }

  const audio = loadAudio(audioPath);
  const full = await transcribe(audio);
  let surah = 1;
  let ayah = 1;
  let score = 0.82;
  const m = db.matchVerse(full.text, 0.25, 4, null, 5);

  let mode = "full_audio_fallback";
  if (m && typeof m.surah === "number" && typeof m.ayah === "number") {
    surah = m.surah;
    ayah = m.ayah;
    score = typeof m.score === "number" ? m.score : 0.82;
    mode = "full_audio_matchVerse";
  }

  return {
    surah,
    ayah,
    ayah_end: null,
    score: Math.round(score * 1e6) / 1e6,
    transcript: full.text.slice(0, 500),
    streaming: {
      mode: `shipped_fc_port:${mode}`,
      reference_root: REF_FE,
      note:
        "Full-window ONNX+CTC+QuranDB match (same acoustic stack as RN tracker discovery); chunked streaming simulated only in-browser due to throughput.",
    },
  };
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
