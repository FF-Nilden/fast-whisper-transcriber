#!/usr/bin/env python3
"""
================================================================================
  ULTRA-OPTIMIZED LECTURE TRANSCRIPTION ENGINE (Faster-Whisper Turbo + Silero VAD)
  Hardware Target: AMD Ryzen 7 PRO 7840U / Zen 4 AVX-512 / Windows 11
================================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ottimizzazione OpenMP e threading Zen 4 prima dei caricamenti C++
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CT2_USE_EXPERIMENTAL_PACKED_GEMM"] = "1"

import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Formatta i secondi in timestamp HH:MM:SS,mmm (SRT) o HH:MM:SS.mmm (VTT)"""
    millis = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    sep = "," if srt_format else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"

def save_outputs(base_path: Path, segments_data: list, total_duration: float, elapsed_time: float):
    """Salva le trascrizioni nei formati TXT, SRT, VTT e JSON strutturato"""
    txt_path = base_path.with_suffix(".txt")
    srt_path = base_path.with_suffix(".srt")
    vtt_path = base_path.with_suffix(".vtt")
    json_path = base_path.with_suffix(".json")

    # 1. Testo Semplice (TXT)
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments_data:
            speaker_tag = f"[{seg['speaker']}] " if "speaker" in seg else ""
            f.write(f"[{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}] {speaker_tag}{seg['text']}\n")

    # 2. Sottotitoli SubRip (SRT)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments_data, 1):
            speaker_tag = f"[{seg['speaker']}] " if "speaker" in seg else ""
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'], srt_format=True)} --> {format_timestamp(seg['end'], srt_format=True)}\n")
            f.write(f"{speaker_tag}{seg['text']}\n\n")

    # 3. Sottotitoli WebVTT (VTT)
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments_data, 1):
            speaker_tag = f"[{seg['speaker']}] " if "speaker" in seg else ""
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{speaker_tag}{seg['text']}\n\n")

    # 4. JSON Completo con Metadati
    full_json = {
        "metadata": {
            "source_file": str(base_path),
            "audio_duration_seconds": round(total_duration, 2),
            "transcription_time_seconds": round(elapsed_time, 2),
            "speedup_realtime": round(total_duration / elapsed_time, 2) if elapsed_time > 0 else 0.0,
            "realtime_factor": round(elapsed_time / total_duration, 4) if total_duration > 0 else 0.0,
            "total_segments": len(segments_data),
            "total_words": sum(len(s["text"].split()) for s in segments_data)
        },
        "segments": segments_data
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)

    return txt_path, srt_path, vtt_path, json_path

def run_diarization(audio_path: str, sherpa_model_dir: str):
    """Esegue la Diarizzazione degli interlocutori via Sherpa-ONNX PyAnnote se richiesto"""
    try:
        import sherpa_onnx
        seg_model = os.path.join(sherpa_model_dir, "pyannote_diarization", "model.int8.onnx")
        emb_model = os.path.join(sherpa_model_dir, "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
        
        if not (os.path.exists(seg_model) and os.path.exists(emb_model)):
            print("[WARN] Modelli di diarizzazione non trovati. Diarizzazione saltata.")
            return None
        
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model),
                num_threads=2
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=emb_model,
                num_threads=2
            ),
            clustering=sherpa_onnx.FastClusteringConfig(threshold=0.6)
        )
        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        audio_data, sr = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        
        print("  -> Calcolo Diarizzazione interlocutori (PyAnnote INT8)...")
        diar_result = sd.process(audio_data)
        speaker_segments = []
        for item in diar_result:
            speaker_segments.append({
                "start": item.start,
                "end": item.end,
                "speaker": f"Speaker_{item.speaker}"
            })
        return speaker_segments
    except Exception as e:
        print(f"[WARN] Errore durante la diarizzazione: {e}")
        return None

def assign_speakers_to_segments(segments_data: list, diar_segments: list):
    """Assegna l'etichetta dell'interlocutore ad ogni segmento trascritto in base alla sovrapposizione temporale"""
    if not diar_segments:
        return segments_data
    
    for seg in segments_data:
        s_start, s_end = seg["start"], seg["end"]
        best_spk = "Speaker_0"
        max_overlap = 0.0
        
        for d in diar_segments:
            overlap = max(0.0, min(s_end, d["end"]) - max(s_start, d["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = d["speaker"]
        
        seg["speaker"] = best_spk
    return segments_data

def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Lecture Transcription Engine (Faster-Whisper Turbo + Silero VAD)")
    parser.add_argument("audio", type=str, help="Percorso del file audio o video da trascrivere (.wav, .mp3, .m4a, .mp4, ecc.)")
    parser.add_argument("--lang", type=str, default="it", help="Lingua del parlato (default: 'it')")
    parser.add_argument("--threads", type=int, default=2, help="Numero di thread CPU AVX-512 (default: 2 per efficienza termica/batteria)")
    parser.add_argument("--model", type=str, default="large-v3-turbo", help="Modello Faster-Whisper (default: 'large-v3-turbo')")
    parser.add_argument("--diarize", action="store_true", help="Attiva la Speaker Diarization PyAnnote per identificare diversi oratori")
    parser.add_argument("--output_dir", type=str, default=None, help="Cartella di destinazione per i file di output (default: stessa cartella dell'audio)")
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"[ERRORE] File audio non trovato: {audio_path}")
        sys.exit(1)

    print("=" * 80)
    print("      ULTRA-FAST LECTURE TRANSCRIPTION ENGINE")
    print(f"      File Target: {audio_path.name}")
    print(f"      Modello: {args.model} (INT8) | Threads: {args.threads} CPU AVX-512 | Lingua: {args.lang}")
    print("=" * 80)

    # 1. Ispezione Audio
    info_audio = sf.info(str(audio_path))
    duration_s = info_audio.duration
    print(f"\n[1/3] Audio caricato: {duration_s:.1f}s ({duration_s/60:.2f} min) @ {info_audio.samplerate}Hz")

    # 2. Caricamento Modello Faster-Whisper Turbo
    print(f"[2/3] Inizializzazione motore C++ (CTranslate2 INT8, {args.threads} thread)...")
    t0_load = time.perf_counter()
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=args.threads,
        num_workers=1
    )
    print(f"      Motore pronto in {time.perf_counter() - t0_load:.2f}s")

    # 3. Trascrizione con Silero VAD Integrato
    print("\n[3/3] Trascrizione in corso con Silero VAD (Zero-Latency Sentence Segmentation)...")
    t_start = time.perf_counter()
    
    segments, info = model.transcribe(
        str(audio_path),
        language=args.lang,
        beam_size=1, # Greedy decoding: ultra-veloce e preciso
        vad_filter=True, # Silero VAD C++ integrato
        vad_parameters=dict(
            min_silence_duration_ms=500, # 500ms di pausa chiudono il periodo
            speech_pad_ms=200,          # 200ms di padding per non tagliare le consonanti
            threshold=0.5               # Soglia di confidenza voce
        ),
        temperature=0.0,
        condition_on_previous_text=False # Previene allucinazioni a ciclo continuo
    )

    segments_data = []
    print("\n--- INIZIO TRASCRIZIONE ---")
    for s in segments:
        text_clean = s.text.strip()
        if text_clean:
            segments_data.append({
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": text_clean
            })
            print(f"[{format_timestamp(s.start)} --> {format_timestamp(s.end)}] {text_clean}")
    
    elapsed_time = time.perf_counter() - t_start
    print("--- FINE TRASCRIZIONE ---\n")

    # 4. Diarizzazione Opzionale
    if args.diarize:
        sherpa_dir = os.path.join(os.path.dirname(__file__), "models", "sherpa")
        diar_segments = run_diarization(str(audio_path), sherpa_dir)
        if diar_segments:
            segments_data = assign_speakers_to_segments(segments_data, diar_segments)
            print("  -> Interlocutori associati con successo ai segmenti!")

    # 5. Salvataggio Output
    out_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_out = out_dir / audio_path.stem

    txt_p, srt_p, vtt_p, json_p = save_outputs(base_out, segments_data, duration_s, elapsed_time)

    # 6. Report Prestazionale
    speedup = duration_s / elapsed_time if elapsed_time > 0 else 0.0
    rtf = elapsed_time / duration_s if duration_s > 0 else 0.0
    words_count = sum(len(s["text"].split()) for s in segments_data)

    print("=" * 80)
    print("                      RIEPILOGO PRESTAZIONI")
    print("=" * 80)
    print(f"Durata Audio Originale:     {duration_s:.2f} s ({duration_s/60:.2f} min)")
    print(f"Tempo Totale Trascrizione:  {elapsed_time:.2f} s ({elapsed_time/60:.2f} min)")
    print(f"Velocità di Calcolo:        {speedup:.2f}x Real-Time (RTF: {rtf:.4f})")
    print(f"Parole Trascritte:          {words_count} su {len(segments_data)} segmenti")
    print("-" * 80)
    print("File Generati con Successo:")
    print(f"  • Testo (TXT):            {txt_p}")
    print(f"  • Sottotitoli (SRT):      {srt_p}")
    print(f"  • Sottotitoli Web (VTT):  {vtt_p}")
    print(f"  • Dati & Metadati (JSON): {json_p}")
    print("=" * 80)

if __name__ == "__main__":
    main()
