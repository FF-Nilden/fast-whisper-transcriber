# 🎙️ Whisper Audio Intelligence Engine (AMD Zen 4 AVX-512 Optimized)

Pipeline di trascrizione audio e sottotitolazione ad altissime prestazioni per lezioni universitarie e parlato in lingua italiana, ottimizzata per **AMD Ryzen 7 PRO 7840U / Zen 4 con istruzioni hardware AVX-512 VNNI**.

---

## ⚡ Caratteristiche Principali

- 🚀 **Velocità 2.1x Real-Time a 2 Thread:** Trascrive 3 minuti di lezione in 85 secondi (o 30 minuti in ~14 minuti) impegnando solo 2 core logici della CPU e lasciando i rimanenti 6 core in modalità *Deep Sleep* per la massima autonomia della batteria.
- 🎯 **Silero VAD C++ Integrato (128 µs di latenza):** Riconosce ed elimina automaticamente silenzi e pause d'aula, prevenendo le allucinazioni e riducendo il tempo di inferenza.
- 🇮🇹 **Comprensione Accademica Perfetta:** Gestisce senza errori termini tecnici, formule di fisica, ingegneria, matematica e punteggiatura naturale.
- 👥 **Speaker Diarization Opzionale:** Riconoscimento e separazione automatica degli interlocutori (Docente vs Studenti) tramite PyAnnote ONNX.
- 📄 **Esportazione Multi-Formato:** Genera simultaneamente formati `.txt`, `.srt` (VLC/YouTube), `.vtt` (Web) e `.json` strutturato.

---

## 🚀 Utilizzo Rapido

### 1. Trascrizione Standard Ottimizzata (Consigliata)
```bash
python transcribe.py sample_audio/lezione_universitaria_16k.wav
```

### 2. Trascrizione con Identificazione Parlanti (Speaker Diarization)
```bash
python transcribe.py audio_lezione.mp3 --diarize
```

### 3. Trascrizione Massima Velocità su Presa di Corrente (4 Thread)
```bash
python transcribe.py audio_lezione.m4a --threads 4
```

---

## 📊 Documentazione Scientifica e Benchmark

Per consultare l'intero resoconto con tabelle comparative, analisi hardware (CPU vs iGPU vs NPU), studio di ablazione dei filtri audio e scaling dei thread:
👉 Leggi il file [`Resoconto_Benchmark_Lezione.md`](file:///c:/Users/franc/Desktop/Esperimenti/Test_Whisper/Resoconto_Benchmark_Lezione.md)

I dati grezzi dei benchmark sono conservati nella cartella [`benchmarks_data/`](file:///c:/Users/franc/Desktop/Esperimenti/Test_Whisper/benchmarks_data).
