Index.html  
```
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مشروع صقر | SAQR - كشف الأصوات المزيفة</title>
    <!-- استيراد خط عربي جميل من جوجل -->
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b1329;
            --card-bg: #131c38;
            --accent-color: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #1e293b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Cairo', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* شريط التنقل */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 8%;
            border-bottom: 1px solid var(--border-color);
            background-color: rgba(11, 19, 41, 0.8);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo span {
            color: var(--text-main);
        }

        nav ul {
            display: flex;
            gap: 25px;
            list-style: none;
        }

        nav ul li a {
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s;
        }

        nav ul li a:hover {
            color: var(--accent-color);
        }

        /* المحتوى الرئيسي */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
            text-align: center;
        }

        .hero-title {
            font-size: 2.5rem;
            margin-bottom: 15px;
            font-weight: 700;
        }

        .hero-title span {
            color: var(--accent-color);
        }

        .hero-subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 600px;
            margin-bottom: 40px;
            line-height: 1.6;
        }

        /* صندوق الرفع */
        .upload-container {
            background-color: var(--card-bg);
            border: 2px dashed var(--border-color);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 650px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .upload-container:hover {
            border-color: var(--accent-color);
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            color: var(--accent-color);
        }

        .upload-text {
            margin-bottom: 20px;
            color: var(--text-muted);
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background-color: var(--accent-color);
            color: var(--bg-color);
            padding: 12px 30px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            border: none;
            transition: background 0.3s;
            font-size: 1rem;
        }

        .upload-btn:hover {
            background-color: var(--accent-hover);
        }

        /* زر الفحص والتنفيذ */
        .analyze-btn {
            margin-top: 25px;
            background-color: #10b981;
            color: white;
            padding: 12px 40px;
            border-radius: 8px;
            font-weight: 700;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            transition: background 0.3s;
            display: none; /* يظهر لما يتم اختيار ملف */
        }

        .analyze-btn:hover {
            background-color: #059669;
        }

        /* قسم النتائج (افتراضياً مخفي) */
        .results-section {
            display: none;
            margin-top: 40px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            width: 100%;
            max-width: 650px;
            text-align: right;
        }

        .results-section h3 {
            margin-bottom: 15px;
            color: var(--accent-color);
        }

        .result-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
        }

        /* التذييل */
        footer {
            text-align: center;
            padding: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

    <!-- شريط التنقل -->
    <header>
        <div class="logo">🦅 مشروع <span>صقر (SAQR)</span></div>
        <nav>
            <ul>
                <li><a href="#">الرئيسية</a></li>
                <li><a href="#">عن النظام</a></li>
                <li><a href="#">فريق العمل</a></li>
            </ul>
        </nav>
    </header>

    <!-- المحتوى -->
    <main>
        <h1 class="hero-title">نظام كشف الأصوات المزيفة <span>بالذكاء الاصطناعي</span></h1>
        <p class="hero-subtitle">ارفع ملفاً صوتياً لتحليله عبر تحليل المخططات الطيفية (Spectrograms) والتأكد من دقته وأصالته.</p>

        <!-- صندوق رفع الصوت -->
        <div class="upload-container" id="dropArea">
            <div class="upload-icon">🎵</div>
            <p class="upload-text" id="fileNameDisplay">اسحب وأفرغ ملف الصوت هنا، أو اضغط لاختيار ملف</p>
            <input type="file" id="audioFile" class="file-input" accept="audio/*">
            <button class="upload-btn" onclick="document.getElementById('audioFile').click()">اختيار ملف صوتي</button>
            <br>
            <button class="analyze-btn" id="analyzeBtn" onclick="showResults()">بدء التحليل الفوري</button>
        </div>

        <!-- نتائج التحليل -->
        <div class="results-section" id="resultsSection">
            <h3>📊 تقرير نتائج التحليل</h3>
            <div class="result-item">
                <span>حالة الصوت:</span>
                <strong id="statusResult" style="color: #10b981;">صوت حقيقي (Authentic)</strong>
            </div>
            <div class="result-item">
                <span>نسبة الثقة (Confidence):</span>
                <strong id="confidenceResult">98.4%</strong>
            </div>
            <div class="result-item">
                <span>نوع التحليل:</span>
                <span>تحليل المخطط الطيفي (Spectrogram Analysis)</span>
            </div>
        </div>
    </main>

    <!-- التذييل -->
    <footer>
        <p>جميع الحقوق محفوظة &copy; 2026 - مشروع تخرج صقر (SAQR)</p>
    </footer>

    <!-- سكريبت بسيط لتفاعل الزر وتغيير النص عند اختيار ملف -->
    <script>
        const fileInput = document.getElementById('audioFile');
        const fileNameDisplay = document.getElementById('fileNameDisplay');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultsSection = document.getElementById('resultsSection');

        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = "الملف المختار: " + fileInput.files[0].name;
                analyzeBtn.style.display = "inline-block";
                resultsSection.style.display = "none"; // إخفاء النتائج السابقة لو غيرت الملف
            }
        });

        function showResults() {
            // محاكاة لعملية التحليل (لاحقاً بتربطونها بالـ Python والـ Backend)
            analyzeBtn.textContent = "جاري التحليل...";
            setTimeout(() => {
                analyzeBtn.textContent = "بدء التحليل الفوري";
                resultsSection.style.display = "block";
            }, 1000);
        }
    </script>

</body>
</html>
```
