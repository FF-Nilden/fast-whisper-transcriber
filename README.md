# Trascrizione audio in italiano (Faster-Whisper)

Script da riga di comando per trascrivere audio in italiano su CPU, pensato per
registrazioni con un singolo relatore (lezioni, conferenze, interviste).
Genera testo con timestamp e sottotitoli.

Una sola dipendenza: `faster-whisper`.

---

## Installazione

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Il modello viene scaricato automaticamente al primo avvio (`large-v3-turbo`,
circa 800 MB) e riutilizzato dalla cache locale nelle esecuzioni successive.

---

## Uso

```bash
# Trascrizione standard
python transcribe.py lezione.wav

# Privilegiando l'accuratezza invece della velocità
python transcribe.py lezione.m4a --preset qualita

# Con più thread
python transcribe.py lezione.mp3 --threads 4

# Salvando altrove
python transcribe.py lezione.wav --output_dir ./trascrizioni
```

Per ogni file di ingresso vengono generati quattro file:

| File | Contenuto |
|---|---|
| `nome.txt` | Testo con timestamp di inizio e fine |
| `nome.srt` | Sottotitoli SubRip (VLC, YouTube) |
| `nome.vtt` | Sottotitoli WebVTT (player web) |
| `nome.json` | Segmenti strutturati e metadati dell'esecuzione |

---

## Opzioni

| Opzione | Default | Descrizione |
|---|---|---|
| `--preset` | `veloce` | `veloce` o `qualita` — vedi sotto |
| `-t`, `--threads` | `2` | Thread CPU |
| `--lang` | `it` | Lingua del parlato |
| `--model` | `large-v3-turbo` | Modello Whisper |
| `--device` | `cpu` | `cpu` o `cuda` |
| `--compute-type` | `int8` | `int8`, `float16`, `float32` |
| `--no-vad` | — | Disattiva il rilevamento del parlato |
| `--output_dir` | cartella dell'audio | Destinazione dei file generati |

I default sono raccolti in cima a `transcribe.py`, nel blocco `CONFIGURAZIONE`,
e si possono modificare lì una volta per tutte.

### I due preset

**`veloce`** usa la decodifica greedy (`beam_size=1`) senza rete di sicurezza:
è rapida ed è la configurazione predefinita.

**`qualita`** usa la beam search a 5 percorsi e riattiva il *fallback di
temperatura*, il meccanismo che fa ritentare a Whisper un segmento quando la
decodifica dà segni di cedimento. È più lenta ma sbaglia meno, soprattutto su
audio riverberato, termini tecnici e nomi propri.

**Quanto più lenta e quanto più accurata? Non è stato misurato.** Vedi la nota
sui benchmark in fondo.

### Nota sui modelli

`large-v3-turbo` è multilingua. I modelli `distil-*`, spesso consigliati come
alternativa leggera, sono **solo inglese** e non funzionano in italiano.
Le alternative valide sono `medium` (più lento) e `small` (più leggero ma meno
preciso su acronimi e termini tecnici).

---

## Come funziona

```
audio → Silero VAD → Faster-Whisper (CTranslate2, INT8) → txt / srt / vtt / json
```

Il **VAD** (Voice Activity Detection, integrato in faster-whisper) individua ed
esclude le pause prima di passare l'audio al modello. Riduce il tempo di calcolo
e soprattutto previene le allucinazioni, cioè il testo inventato che Whisper
tende a produrre sui silenzi prolungati.

`condition_on_previous_text` è disattivato di proposito: impedisce a Whisper di
avvitarsi in loop di frasi ripetute, al costo di un po' di contesto fra un
segmento e il successivo.

---

## Formati audio

Whisper decodifica autonomamente tramite PyAV, incluso nel pacchetto: WAV, MP3,
M4A, MP4, FLAC, OGG, AAC e altri. Non serve installare FFmpeg separatamente.

---

## Prestazioni e limiti noti

Su un portatile AMD Ryzen 7 PRO 7840U, con `large-v3-turbo` in INT8 a 2 thread,
l'ordine di grandezza osservato è **circa 2x real-time**: tre minuti di audio
trascritti in un minuto e mezzo, con ventole silenziose e consumo contenuto.

**Questo numero va però considerato preliminare.** I benchmark della versione
precedente non erano controllati: due misurazioni della stessa identica
configurazione differivano di un fattore 4, quasi certamente per throttling
termico durante run consecutivi. Le raccomandazioni sul numero ottimale di
thread che ne erano state ricavate non sono attendibili, tanto più che erano
falsate da un bug di threading ora corretto (le variabili OpenMP restavano
fissate a 2 anche quando si chiedevano più thread).

**L'accuratezza non è mai stata misurata.** Non esiste una trascrizione di
riferimento né un calcolo del WER: le valutazioni sulla qualità erano giudizi a
occhio. Lo script `benchmark.py` è predisposto per entrambe le misure quando si
vorrà farle; le istruzioni sono nella sua intestazione.

### Cosa questo programma non fa

- **Non distingue i parlanti.** Un modulo di diarizzazione era stato tentato e
  poi rimosso: pesava troppo per il beneficio, e nell'unico test disponibile
  aveva riconosciuto un solo interlocutore. Per registrazioni con più voci da
  separare serve un altro strumento.
- **Non gira su NPU.** L'esecuzione su acceleratore neurale era stata provata e
  abbandonata: l'architettura XDNA di prima generazione non gestisce
  efficacemente modelli encoder-decoder con cache dinamica come Whisper.
- **Non lavora in streaming.** Elabora file già registrati, non audio dal vivo.

---

## Struttura

```
transcribe.py       Motore di trascrizione
benchmark.py        Script di misurazione (non ancora eseguito)
requirements.txt    Dipendenze
sample_audio/       Campione di prova e output di esempio
benchmarks_data/    Dati grezzi delle misurazioni precedenti (vedi avvertenza sopra)
```
