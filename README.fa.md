# AdaptiveMetric RAG

**هر Query باید تعریف خودش را از شباهت داشته باشد.**

AdaptiveMetric RAG یک دستیار دانش self-hosted و چندزبانه است که روش retrieval را برای هر سؤال تغییر می‌دهد. سؤال زمانی به تاریخ وزن بیشتری می‌دهد، سؤال factual روی entity و عبارت دقیق تمرکز می‌کند و سؤال مفهومی semantic similarity را مهم‌تر می‌داند. تمام پاسخ‌ها citation قابل بررسی دارند.

[English documentation](README.md)

## قابلیت‌ها

- Query Analyzer برای intentهای factual، conceptual، causal، numeric، temporal و technical/code
- ترکیب پویا از Dense، BM25، Entity، Number، Time و Metadata
- مسیر سه‌مرحله‌ای candidate retrieval، adaptive scoring و grounded generation
- confidence score و سیگنال early exit
- پشتیبانی از PDF، DOCX، TXT، Markdown، CSV، JSON و HTML
- citation درون پاسخ همراه نام فایل، شماره صفحه، excerpt، chunk ID و score
- حالت Local بدون API key و پشتیبانی از Ollama، OpenAI-compatible/dedicated و Gemini
- انتخاب embedding محلی یا مدل نصب‌شده در Ollama همراه re-index خودکار هنگام تغییر مدل
- نگهداری پایدار اسناد، تنظیمات و گفتگوها در SQLite
- رابط responsive و فارسی با کتابخانه منابع و نمایش metricهای retrieval
- نمایش زیبای Markdown پاسخ‌ها همراه تم روشن و تیره پایدار
- خروجی JSON مستقل برای هر گفتگو شامل پیام‌ها، Citationها و مشخصات امن مدل‌ها
- اجرای Docker با یک دستور روی پورت `2266`

## معماری

```text
Query → Query Analyzer → Metric Router
                            │
          Dense + BM25 + Entity + Number + Time + Metadata
                            │
                       Top candidates
                            │
                  Adaptive metric scoring
                            │
                  Confidence / Early exit
                            │
                     Grounded context
                            │
             Local / Ollama / OpenAI / Gemini
                            │
                      Answer + Citation
```

Embedding داخلی یک feature hashing چندزبانه و قطعی با ۳۸۴ بُعد است. بدون دانلود مدل، آفلاین و فوری اجرا می‌شود و برای knowledge baseهای شخصی کوچک و متوسط مناسب است. برای مقیاس چند میلیون chunk می‌توانید candidate selection در `app/retrieval.py` را با FAISS یا Qdrant جایگزین کنید و adaptive scoring را روی Top 50–200 نگه دارید.

## اجرای سریع با Docker

```bash
git clone <your-repository-url>
cd AdaptiveMetric-RAG
docker compose up --build -d
```

سپس **http://localhost:2266** را باز کنید. Provider پیش‌فرض Local است و به کلید نیاز ندارد.

برای توقف:

```bash
docker compose down
```

داده‌ها در volume با نام `adaptive_rag_data` ذخیره می‌شوند و با ساخت مجدد container از بین نمی‌روند.

## تنظیم Provider

از داخل رابط به **تنظیمات ← مدل پاسخ‌گو** بروید.

مدل پاسخ‌گو و مدل embedding دو تنظیم مستقل‌اند. در **تنظیمات ← Retrieval** می‌توانید بردارساز محلی ۳۸۴ بُعدی یا یکی از مدل‌های embedding نصب‌شده در Ollama را انتخاب کنید. رابط فهرست مدل‌های Ollama را دریافت می‌کند و پیش از فعال‌کردن مدل جدید، تمام chunkهای موجود را خودکار دوباره embedding می‌کند.

### Ollama

Ollama را روی سیستم میزبان اجرا و یک مدل دریافت کنید:

```bash
ollama pull llama3.2
```

سپس Provider را روی Ollama، مدل را `llama3.2` و Base URL را روی `http://host.docker.internal:11434` قرار دهید.

### OpenAI یا Dedicated Endpoint

گزینه **OpenAI / Dedicated compatible** را انتخاب و نام مدل، API Key و Base URL را وارد کنید. مقدار پیش‌فرض URL برابر `https://api.openai.com/v1` است. هر endpoint اختصاصی سازگار با `POST /chat/completions` مثل vLLM، LM Studio یا gateway سازمانی نیز قابل استفاده است.

### Gemini

گزینه Google Gemini را انتخاب کنید، مدلی مثل `gemini-2.5-flash` و API Key را قرار دهید.

برای OpenAI و Gemini می‌توانید کلیدها را به‌ترتیب با `OPENAI_API_KEY` و `GEMINI_API_KEY` در فایل `.env` نیز تعریف کنید. کلید ثبت‌شده در UI سمت سرور ذخیره می‌شود و هرگز به مرورگر بازگردانده نمی‌شود. در استقرار عمومی از secret manager، TLS و لایه authentication استفاده کنید.

## تنظیمات Retrieval

| تنظیم | پیش‌فرض | کاربرد |
|---|---:|---|
| Candidate pool | 100 | تعداد candidateهای ترکیبی Dense/BM25 که adaptive metric بررسی می‌کند |
| Context chunks | 5 | تعداد منبع ارسالی به مدل پاسخ‌گو |
| Chunk size | 900 | طول تقریبی هر chunk برحسب کاراکتر |
| Chunk overlap | 140 | هم‌پوشانی chunkهای مجاور |
| Confidence threshold | 0.58 | آستانه قابل تنظیم برای flowهای confidence-aware |
| Early exit | فعال | علامت‌گذاری retrievalهای قطعی برای رد کردن مراحل سنگین اختیاری |
| Query expansion | فعال | فعال‌سازی گسترش جستجو در توسعه‌های confidence-aware |

تنظیمات chunking روی فایل‌هایی که بعداً upload می‌شوند اعمال خواهد شد؛ برای اعمال روی اسناد قبلی باید آن‌ها را دوباره اضافه کنید.

## اجرای توسعه‌ای

Python 3.11 یا جدیدتر لازم است.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 2266
```

تست‌ها:

```bash
pytest -q
```

مستندات API در `http://localhost:2266/docs` در دسترس است.

## نحوه Citation

پیش از generation، chunkهای منتخب شماره‌گذاری می‌شوند و مدل باید claimها را با قالب `[1]` و `[2]` ارجاع دهد. API مستقل از فرمت متن مدل، آرایه ساختاریافته `citations` را هم برمی‌گرداند؛ بنابراین UI همیشه می‌تواند نام سند، صفحه، excerpt، chunk ID و adaptive score را نمایش دهد.

## نکات Production

- پیش از انتشار عمومی، authentication و reverse proxy امن اضافه کنید.
- SQLite برای deployment تک‌سرور ساده است؛ برای نوشتن هم‌زمان چندکاربره PostgreSQL مناسب‌تر است.
- برای مجموعه بسیار بزرگ، مرحله اول ANN را با Qdrant یا FAISS اجرا کنید و Adaptive Metric را روی Top 50–200 نگه دارید.
- PDF اسکن‌شده به OCR نیاز دارد؛ `pypdf` به‌تنهایی OCR انجام نمی‌دهد.
- کیفیت را روی داده برچسب‌خورده خودتان با Recall@K، MRR، nDCG و p95 latency بسنجید.

## License

پیش از انتشار عمومی، License متناسب با نحوه توزیع پروژه را اضافه کنید.
