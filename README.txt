=========================================
Extragere și Transcriere Voci din Video
=========================================

1. DESCRIERE
Acest proiect extrage componenta audio dintr-un fișier video, izolează vocea (vocals) de restul sunetelor folosind modelul Demucs, identifică segmentele de vorbire cu Silero VAD, le transcrie folosind modelul OpenAI Whisper și generează o spectrogramă comparativă.

2. DATE DE INTRARE (DATASET)
Videoclipul sursă trebuie plasat în folderul `data/` și trebuie să se numească `input_video.mp4`.
[NOTĂ IMPORTANTĂ DACA VIDEOUL E PREA MARE: Deoarece fișierul video original este prea mare pentru a fi inclus în arhivă, acesta poate fi descărcat de la următorul link: (INSEREAZĂ LINK DE GOOGLE DRIVE/YOUTUBE AICI). Descărcați-l și plasați-l în folderul /data înainte de rulare.]

3. CERINȚE PREALABILE (INSTALARE)
Asigurați-vă că aveți instalat Python 3.8+ și un mediu virtual activat. Instalați pachetele necesare rulând comanda:

pip install -r requirements.txt

(Notă: Pentru rularea optimă a modelelor AI, sistemul va folosi automat CUDA dacă o placă video NVIDIA compatibilă este detectată, altfel va face fallback pe CPU).

4. RULAREA EXPERIMENTULUI
Navigați în directorul principal al proiectului (unde se află main.py) și rulați în consolă:

python main.py

5. REZULTATE AȘTEPTATE
- La finalul execuției, se va crea un director `vocal_segments/` care va conține fragmentele audio (.wav) feliate doar cu vocile extrase.
- Se va genera fișierul `transcriere.txt` cu timestamp-urile și textul aferent.
- Va apărea o fereastră cu spectrogramele comparative (Mixajul original vs Vocea extrasă integral).