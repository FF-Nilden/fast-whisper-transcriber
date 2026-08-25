#!/usr/bin/env python3
"""
================================================================================
  MOTORE DI TRASCRIZIONE AUDIO IN ITALIANO (Faster-Whisper + Silero VAD)
  CPU-only, dipendenza singola: faster-whisper.
================================================================================

Uso rapido:
    python transcribe.py lezione.wav
    python transcribe.py lezione.m4a --preset qualita
    python transcribe.py lezione.mp3 --threads 4

Genera automaticamente: .txt  .srt  .vtt  .json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# ==============================================================================
#  CONFIGURAZIONE — modifica questi valori per cambiare i default del programma
# ==============================================================================

THREADS_DEFAULT = 2              # Thread CPU. 2 = silenzioso e parco di batteria.
MODEL_DEFAULT = "large-v3-turbo"  # Multilingua. NON usare i modelli 'distil-*': solo inglese.
LANG_DEFAULT = "it"
PRESET_DEFAULT = "veloce"         # "veloce" oppure "qualita"
DEVICE_DEFAULT = "cpu"
COMPUTE_TYPE_DEFAULT = "int8"     # su GPU NVIDIA: "float16"

# Parametri del Voice Activity Detection (Silero, integrato in faster-whisper).
VAD_PARAMS = dict(
    min_silence_duration_ms=500,  # pausa che chiude un periodo
    speech_pad_ms=200,            # padding per non troncare le consonanti
    threshold=0.5,                # soglia di confidenza voce
)

# I due profili di decodifica.
#   veloce  : decodifica greedy, nessun fallback. Rapida, qualche errore in più.
#   qualita : beam search a 5 + fallback di temperatura (rete di sicurezza che
#             ritenta un segmento quando la decodifica va male). Più lenta.
PRESETS = {
    "veloce": dict(
        beam_size=1,
        temperature=0.0,
    ),
    "qualita": dict(
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    ),
}

# ==============================================================================
#  Threading — DEVE essere impostato PRIMA di importare faster_whisper, perché il
#  runtime C++ legge queste variabili al caricamento. Per questo facciamo una
#  lettura anticipata di --threads da sys.argv invece di aspettare argparse.
# ==============================================================================


def _leggi_threads_da_argv(argv, default: int) -> int:
    """Estrae il valore di --threads/-t da sys.argv prima che argparse esista."""
    for i, arg in enumerate(argv):
        valore = None
        if arg in ("--threads", "-t") and i + 1 < len(argv):
            valore = argv[i + 1]
        elif arg.startswith("--threads="):
            valore = arg.split("=", 1)[1]
        if valore is not None:
            try:
                return max(1, int(valore))
            except ValueError:
                return default
    return default


_THREADS = _leggi_threads_da_argv(sys.argv[1:], THREADS_DEFAULT)
os.environ["OMP_NUM_THREADS"] = str(_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_THREADS)

from faster_whisper import WhisperModel  # noqa: E402  (import dopo le env var)


# ==============================================================================
#  Output
# ==============================================================================


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Formatta i secondi come HH:MM:SS,mmm (SRT) oppure HH:MM:SS.mmm (VTT)."""
    millis = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    sep = "," if srt_format else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def save_outputs(base_path: Path, segments_data: list, metadata: dict):
    """Salva la trascrizione in TXT, SRT, VTT e JSON."""
    txt_path = base_path.with_suffix(".txt")
    srt_path = base_path.with_suffix(".srt")
    vtt_path = base_path.with_suffix(".vtt")
    json_path = base_path.with_suffix(".json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments_data:
            inizio = format_timestamp(seg["start"])
            fine = format_timestamp(seg["end"])
            f.write(f"[{inizio} --> {fine}] {seg['text']}\n")

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments_data, 1):
            inizio = format_timestamp(seg["start"], srt_format=True)
            fine = format_timestamp(seg["end"], srt_format=True)
            f.write(f"{i}\n{inizio} --> {fine}\n{seg['text']}\n\n")

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments_data:
            inizio = format_timestamp(seg["start"])
            fine = format_timestamp(seg["end"])
            f.write(f"{inizio} --> {fine}\n{seg['text']}\n\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": metadata, "segments": segments_data},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return txt_path, srt_path, vtt_path, json_path


# ==============================================================================
#  Programma principale
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Trascrizione audio in italiano con Faster-Whisper e Silero VAD."
    )
    parser.add_argument("audio", type=str,
                        help="File audio o video da trascrivere (.wav, .mp3, .m4a, .mp4, .flac, ...)")
    parser.add_argument("--lang", type=str, default=LANG_DEFAULT,
                        help=f"Lingua del parlato (default: '{LANG_DEFAULT}')")
    parser.add_argument("-t", "--threads", type=int, default=THREADS_DEFAULT,
                        help=f"Thread CPU da impiegare (default: {THREADS_DEFAULT})")
    parser.add_argument("--preset", type=str, default=PRESET_DEFAULT,
                        choices=sorted(PRESETS.keys()),
                        help=f"Profilo di decodifica (default: '{PRESET_DEFAULT}')")
    parser.add_argument("--model", type=str, default=MODEL_DEFAULT,
                        help=f"Modello Faster-Whisper (default: '{MODEL_DEFAULT}')")
    parser.add_argument("--device", type=str, default=DEVICE_DEFAULT,
                        help=f"Dispositivo di calcolo: cpu | cuda (default: '{DEVICE_DEFAULT}')")
    parser.add_argument("--compute-type", dest="compute_type", type=str,
                        default=COMPUTE_TYPE_DEFAULT,
                        help=f"Quantizzazione: int8 | float16 | float32 (default: '{COMPUTE_TYPE_DEFAULT}')")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disattiva il Voice Activity Detection (sconsigliato)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Cartella di destinazione (default: quella dell'audio)")

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"[ERRORE] File audio non trovato: {audio_path}")
        sys.exit(1)

    # Coerenza fra il valore letto in anticipo e quello di argparse: se l'utente
    # ha usato una sintassi che la lettura anticipata non copre, avvisiamo invece
    # di far girare il programma in una configurazione di threading incoerente.
    if args.threads != _THREADS:
        print(f"[AVVISO] Threading impostato a {_THREADS} all'avvio del runtime C++, "
              f"ma --threads vale {args.threads}. Uso {_THREADS} per coerenza.")
    threads = _THREADS

    opzioni_preset = PRESETS[args.preset]

    print("=" * 78)
    print("      TRASCRIZIONE AUDIO — Faster-Whisper")
    print(f"      File:    {audio_path.name}")
    print(f"      Modello: {args.model} ({args.compute_type}) su {args.device}")
    print(f"      Thread:  {threads}   Lingua: {args.lang}   Preset: {args.preset}")
    print("=" * 78)

    print(f"\n[1/3] Caricamento del modello...")
    t0 = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=threads,
        num_workers=1,
    )
    print(f"      Modello pronto in {time.perf_counter() - t0:.2f}s")

    print(f"\n[2/3] Trascrizione in corso"
          f"{'' if args.no_vad else ' (con Silero VAD)'}...")
    t_start = time.perf_counter()

    segments, info = model.transcribe(
        str(audio_path),
        language=args.lang,
        vad_filter=not args.no_vad,
        vad_parameters=None if args.no_vad else dict(VAD_PARAMS),
        condition_on_previous_text=False,  # previene i loop di frasi ripetute
        **opzioni_preset,
    )

    duration_s = info.duration
    duration_vad = getattr(info, "duration_after_vad", None)

    segments_data = []
    print("\n--- INIZIO TRASCRIZIONE ---")
    for s in segments:
        testo = s.text.strip()
        if testo:
            segments_data.append({
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": testo,
            })
            print(f"[{format_timestamp(s.start)} --> {format_timestamp(s.end)}] {testo}")
    elapsed = time.perf_counter() - t_start
    print("--- FINE TRASCRIZIONE ---\n")

    if not segments_data:
        print("[AVVISO] Nessun parlato riconosciuto. "
              "Se il file contiene voce, prova con --no-vad.")

    parole = sum(len(s["text"].split()) for s in segments_data)
    metadata = {
        "source_file": audio_path.name,
        "model": args.model,
        "compute_type": args.compute_type,
        "device": args.device,
        "threads": threads,
        "preset": args.preset,
        "language": args.lang,
        "vad": not args.no_vad,
        "audio_duration_seconds": round(duration_s, 2),
        "audio_duration_after_vad_seconds": (
            round(duration_vad, 2) if duration_vad is not None else None
        ),
        "transcription_time_seconds": round(elapsed, 2),
        "speedup_realtime": round(duration_s / elapsed, 2) if elapsed > 0 else 0.0,
        "realtime_factor": round(elapsed / duration_s, 4) if duration_s > 0 else 0.0,
        "total_segments": len(segments_data),
        "total_words": parole,
    }

    print("[3/3] Salvataggio dei file di output...")
    out_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_p, srt_p, vtt_p, json_p = save_outputs(
        out_dir / audio_path.stem, segments_data, metadata
    )

    print("\n" + "=" * 78)
    print("                      RIEPILOGO")
    print("=" * 78)
    print(f"Durata audio:          {duration_s:.2f} s ({duration_s / 60:.2f} min)")
    if duration_vad is not None:
        tagliato = duration_s - duration_vad
        print(f"Silenzio tagliato:     {tagliato:.2f} s "
              f"({tagliato / duration_s * 100:.1f}% dell'audio)")
    print(f"Tempo di trascrizione: {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print(f"Velocità:              {metadata['speedup_realtime']:.2f}x real-time "
          f"(RTF {metadata['realtime_factor']:.4f})")
    print(f"Risultato:             {parole} parole in {len(segments_data)} segmenti")
    print("-" * 78)
    print("File generati:")
    for etichetta, percorso in (("Testo (TXT)", txt_p), ("Sottotitoli (SRT)", srt_p),
                                ("Sottotitoli web (VTT)", vtt_p), ("Dati (JSON)", json_p)):
        print(f"  • {etichetta:<22} {percorso}")
    print("=" * 78)


if __name__ == "__main__":
    main()
