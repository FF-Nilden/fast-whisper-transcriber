#!/usr/bin/env python3
"""
================================================================================
  SCRIPT DI BENCHMARK CONTROLLATO
================================================================================

Misura le prestazioni di trascrizione applicando tre accorgimenti che rendono i
numeri confrontabili fra loro:

  1. PAUSA DI RAFFREDDAMENTO fra un run e l'altro (default 180 s). Senza questa,
     su un SoC da 15-28 W il throttling termico falsa completamente i risultati:
     è la spiegazione più probabile delle incongruenze nei benchmark precedenti.
  2. RIPETIZIONI con calcolo della MEDIANA (default 3). Un singolo run non dice
     nulla: basta un processo in background per raddoppiare i tempi.
  3. ALIMENTAZIONE DICHIARATA (--alimentazione rete|batteria), registrata nel
     file dei risultati. Run a batteria e run collegati alla rete non vanno
     MAI confrontati fra loro.

Ogni configurazione gira in un PROCESSO SEPARATO, perché le variabili di
threading OpenMP vengono lette una sola volta al caricamento del runtime C++ e
non possono essere cambiate a caldo dentro lo stesso processo.

--------------------------------------------------------------------------------
ESEMPI D'USO
--------------------------------------------------------------------------------

  # Confronto sui thread (la domanda aperta: 2 vs 4 dopo la correzione del bug)
  python benchmark.py audio.wav --threads 1 2 4 8 --alimentazione rete

  # Confronto fra i due preset di decodifica
  python benchmark.py audio.wav --threads 2 --preset veloce qualita

  # Prova rapida senza attese (risultati NON confrontabili, solo per collaudo)
  python benchmark.py audio.wav --threads 2 4 --ripetizioni 1 --cooldown 0

--------------------------------------------------------------------------------
CALCOLO DEL WER (accuratezza reale)
--------------------------------------------------------------------------------

I tempi dicono quanto è veloce, non quanto è corretto. Per misurare gli errori
serve una trascrizione di riferimento corretta a mano UNA volta sola:

    pip install jiwer
    python benchmark.py audio.wav --threads 2 --riferimento riferimento.txt

Il riferimento è un file .txt di solo testo, senza timestamp. Il confronto viene
normalizzato (minuscole, senza punteggiatura) prima del calcolo.

================================================================================
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_TRASCRIZIONE = Path(__file__).parent / "transcribe.py"


def normalizza(testo: str) -> str:
    """Minuscole, via la punteggiatura, spazi compattati."""
    testo = testo.lower()
    testo = re.sub(r"[^\w\sàèéìòùáíóúü']", " ", testo)
    return re.sub(r"\s+", " ", testo).strip()


def testo_da_json(percorso_json: Path) -> str:
    dati = json.loads(percorso_json.read_text(encoding="utf-8"))
    return " ".join(s["text"] for s in dati["segments"])


def calcola_wer(riferimento: str, ipotesi: str):
    """Word Error Rate. Restituisce None se jiwer non è installato."""
    try:
        import jiwer
    except ImportError:
        return None
    return jiwer.wer(normalizza(riferimento), normalizza(ipotesi))


def esegui_run(audio: Path, threads: int, preset: str, modello: str,
               cartella_out: Path) -> dict:
    """Lancia una singola trascrizione in un processo separato."""
    comando = [
        sys.executable, str(SCRIPT_TRASCRIZIONE), str(audio),
        "--threads", str(threads),
        "--preset", preset,
        "--model", modello,
        "--output_dir", str(cartella_out),
    ]
    t0 = time.perf_counter()
    esito = subprocess.run(comando, capture_output=True, text=True)
    durata_processo = time.perf_counter() - t0

    if esito.returncode != 0:
        return {"errore": esito.stderr[-800:] or "processo terminato con errore"}

    percorso_json = cartella_out / f"{audio.stem}.json"
    if not percorso_json.exists():
        return {"errore": "file JSON di output non trovato"}

    meta = json.loads(percorso_json.read_text(encoding="utf-8"))["metadata"]
    return {
        "tempo_trascrizione_s": meta["transcription_time_seconds"],
        "tempo_processo_s": round(durata_processo, 2),
        "velocita_realtime": meta["speedup_realtime"],
        "rtf": meta["realtime_factor"],
        "parole": meta["total_words"],
        "segmenti": meta["total_segments"],
        "testo": testo_da_json(percorso_json),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark controllato per transcribe.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=str, help="File audio di test")
    parser.add_argument("--threads", type=int, nargs="+", default=[2],
                        help="Valori di thread da confrontare (es: --threads 2 4)")
    parser.add_argument("--preset", type=str, nargs="+", default=["veloce"],
                        help="Preset da confrontare (es: --preset veloce qualita)")
    parser.add_argument("--model", type=str, default="large-v3-turbo")
    parser.add_argument("--ripetizioni", type=int, default=3,
                        help="Run per configurazione, si prende la mediana (default: 3)")
    parser.add_argument("--cooldown", type=int, default=180,
                        help="Secondi di pausa fra i run, contro il throttling (default: 180)")
    parser.add_argument("--alimentazione", type=str, required=True,
                        choices=["rete", "batteria"],
                        help="Stato di alimentazione durante il test. OBBLIGATORIO: "
                             "run a batteria e in rete non sono confrontabili.")
    parser.add_argument("--riferimento", type=str, default=None,
                        help="File .txt con la trascrizione corretta, per il WER")
    parser.add_argument("--output", type=str, default="risultati_benchmark.json")

    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    if not audio.exists():
        print(f"[ERRORE] File audio non trovato: {audio}")
        sys.exit(1)

    testo_riferimento = None
    if args.riferimento:
        percorso_rif = Path(args.riferimento)
        if not percorso_rif.exists():
            print(f"[ERRORE] File di riferimento non trovato: {percorso_rif}")
            sys.exit(1)
        testo_riferimento = percorso_rif.read_text(encoding="utf-8")

    cartella_temp = Path("_benchmark_temp")
    cartella_temp.mkdir(exist_ok=True)

    configurazioni = [(t, p) for t in args.threads for p in args.preset]
    totale_run = len(configurazioni) * args.ripetizioni
    attesa_stimata = (totale_run - 1) * args.cooldown / 60

    print("=" * 78)
    print("  BENCHMARK CONTROLLATO")
    print("=" * 78)
    print(f"  Audio:          {audio.name}")
    print(f"  Configurazioni: {len(configurazioni)} x {args.ripetizioni} ripetizioni "
          f"= {totale_run} run")
    print(f"  Cooldown:       {args.cooldown}s  (~{attesa_stimata:.0f} min di sole pause)")
    print(f"  Alimentazione:  {args.alimentazione}")
    if args.cooldown < 60:
        print("\n  [AVVISO] Cooldown sotto i 60s: il throttling termico può falsare")
        print("           i risultati. Usare solo per collaudare lo script.")
    print("=" * 78)

    risultati = []
    contatore = 0

    for threads, preset in configurazioni:
        etichetta = f"{threads} thread / preset {preset}"
        misure = []

        for ripetizione in range(1, args.ripetizioni + 1):
            contatore += 1
            if contatore > 1 and args.cooldown > 0:
                print(f"\n  ...raffreddamento {args.cooldown}s...")
                time.sleep(args.cooldown)

            print(f"\n[{contatore}/{totale_run}] {etichetta} — run {ripetizione}")
            esito = esegui_run(audio, threads, preset, args.model, cartella_temp)

            if "errore" in esito:
                print(f"      FALLITO: {esito['errore']}")
                continue

            print(f"      {esito['tempo_trascrizione_s']:.2f}s "
                  f"({esito['velocita_realtime']:.2f}x real-time), "
                  f"{esito['parole']} parole")
            misure.append(esito)

        if not misure:
            risultati.append({"configurazione": etichetta, "esito": "tutti i run falliti"})
            continue

        tempi = [m["tempo_trascrizione_s"] for m in misure]
        parole = [m["parole"] for m in misure]
        tempo_mediano = statistics.median(tempi)
        run_mediano = min(misure, key=lambda m: abs(m["tempo_trascrizione_s"] - tempo_mediano))

        voce = {
            "configurazione": etichetta,
            "threads": threads,
            "preset": preset,
            "run_validi": len(misure),
            "tempo_mediano_s": round(tempo_mediano, 2),
            "tempo_min_s": round(min(tempi), 2),
            "tempo_max_s": round(max(tempi), 2),
            "dispersione_pct": round((max(tempi) - min(tempi)) / tempo_mediano * 100, 1),
            "velocita_realtime_mediana": round(
                statistics.median([m["velocita_realtime"] for m in misure]), 2),
            "parole_mediana": statistics.median(parole),
            "segmenti": run_mediano["segmenti"],
        }

        if testo_riferimento:
            wer = calcola_wer(testo_riferimento, run_mediano["testo"])
            if wer is None:
                print("      [NOTA] jiwer non installato: WER non calcolato "
                      "(pip install jiwer)")
            else:
                voce["wer"] = round(wer, 4)
                voce["accuratezza_pct"] = round((1 - wer) * 100, 2)
                print(f"      WER: {wer:.2%}  →  accuratezza {(1 - wer):.2%}")

        if voce["dispersione_pct"] > 15:
            voce["avviso"] = ("Dispersione oltre il 15% fra i run: il sistema non era "
                              "in condizioni stabili, dato poco affidabile.")
            print(f"      [AVVISO] Dispersione {voce['dispersione_pct']}% fra i run.")

        risultati.append(voce)

    documento = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "audio": audio.name,
        "modello": args.model,
        "alimentazione": args.alimentazione,
        "ripetizioni": args.ripetizioni,
        "cooldown_s": args.cooldown,
        "riferimento_wer": args.riferimento,
        "risultati": risultati,
    }
    Path(args.output).write_text(
        json.dumps(documento, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("  RIEPILOGO (tempi mediani)")
    print("=" * 78)
    validi = [r for r in risultati if "tempo_mediano_s" in r]
    for r in sorted(validi, key=lambda x: x["tempo_mediano_s"]):
        riga = (f"  {r['configurazione']:<28} {r['tempo_mediano_s']:>8.2f}s   "
                f"{r['velocita_realtime_mediana']:>5.2f}x")
        if "accuratezza_pct" in r:
            riga += f"   accuratezza {r['accuratezza_pct']:.2f}%"
        print(riga)
    print("=" * 78)
    print(f"  Risultati completi salvati in: {args.output}")
    print(f"  File temporanei in: {cartella_temp}/ (eliminabili)")
    print("=" * 78)


if __name__ == "__main__":
    main()
