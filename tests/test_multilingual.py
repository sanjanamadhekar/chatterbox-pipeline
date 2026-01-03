#!/usr/bin/env python3
"""
Comprehensive multilingual stress test suite for ChatterBox TTS
Tests all 23 supported languages with challenging sentences
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatterbox_pipeline import ChatterBoxPipeline


# Challenging test sentences for each language
# Include numbers, punctuation, abbreviations, homographs, and stress patterns
MULTILINGUAL_STRESS_TESTS = {
    "en": {
        "name": "English",
        "tests": [
            "Dr. O'Brien read the lead article at 3:45 PM about bass fishing in 1999.",
            "The tear in her eye appeared as she tried to tear the paper.",
            "WWW.example.com costs $99.99 plus 15% tax. LOL!",
            "Will the invalid's invalid license be close enough to close the door?",
            "The dove dove into the bushes near Route 66.",
            "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
        ]
    },
    "es": {
        "name": "Spanish",
        "tests": [
            "El Dr. García leyó el artículo principal a las 15:45 sobre la pesca en 1999.",
            "¿Cómo se pronuncia 'bourne' en inglés? El número cuesta €99,99 más el 15% de IVA.",
            "WWW.sitio-web.com/página_1 tiene información sobre el año 2024. ¡Órale!",
            "José y María fueron al café a las 7:30 PM con $100 dólares.",
            "¿Dónde está la estación? Está a 3.5 kilómetros de aquí.",
        ]
    },
    "fr": {
        "name": "French",
        "tests": [
            "Dr. Dubois a lu l'article principal à 15h45 sur la pêche en 1999.",
            "Comment prononce-t-on 'through' et 'thorough'? L'adresse coûte 99,99€ plus 15% de TVA.",
            "Les poules du couvent couvent. C'est à www.site-web.fr. MDR!",
            "À 8h30, nous avons rendez-vous au café près de l'église.",
            "Le prix est de 1.234,56€ avec une réduction de 20%.",
        ]
    },
    "de": {
        "name": "German",
        "tests": [
            "Dr. Müller las den Hauptartikel um 15:45 Uhr über das Angeln im Jahr 1999.",
            "Wie spricht man 'squirrel' aus? Die Adresse kostet 99,99€ plus 15% MwSt.",
            "Umfahren ist nicht das Gegenteil von umfahren. WWW.website.de. LOL!",
            "Der Preis beträgt 1.234,56€ mit 20% Rabatt.",
            "Um 7:30 Uhr treffen wir uns am Bahnhof.",
        ]
    },
    "zh": {
        "name": "Chinese",
        "tests": [
            "王博士在下午3点45分阅读了关于1999年钓鱼的主要文章。",
            "这个网址www.网站.com要价99.99美元加15%税。",
            "意大利的意大利面和中国的意面一样吗？哈哈！",
            "2024年1月15日，我在北京见到了李先生。",
            "价格是1,234.56元，打八折后是987.65元。",
        ]
    },
    "ja": {
        "name": "Japanese",
        "tests": [
            "田中博士は1999年の釣りに関する主要記事を午後3時45分に読みました。",
            "アドレスは99.99ドルに15%の税金がかかります。www.サイト.jp",
            "橋の端を箸で叩く。雨と飴、雲と蜘蛛。笑！",
            "2024年1月15日、午後7時30分に東京駅で会いましょう。",
            "価格は1,234円で、20%割引があります。",
        ]
    },
    "ko": {
        "name": "Korean",
        "tests": [
            "박 박사님은 1999년 낚시에 관한 주요 기사를 오후 3시 45분에 읽었습니다.",
            "'Entrepreneur'를 어떻게 발음합니까? 주소는 99.99달러에 15% 세금이 추가됩니다.",
            "간장 공장 공장장은 강 공장장이고 된장 공장 공장장은 장 공장장이다. ㅋㅋㅋ!",
            "2024년 1월 15일 오후 7시 30분에 서울역에서 만나요.",
            "가격은 1,234원이고 20% 할인됩니다.",
        ]
    },
    "it": {
        "name": "Italian",
        "tests": [
            "Il Dr. Rossi ha letto l'articolo principale alle 15:45 sulla pesca nel 1999.",
            "Come si pronuncia 'Worcester'? L'indirizzo costa €99,99 più il 15% di IVA.",
            "Sopra la panca la capra campa, sotto la panca la capra crepa. LOL!",
            "Il prezzo è di 1.234,56€ con uno sconto del 20%.",
            "Ci vediamo alle 19:30 alla stazione di Roma.",
        ]
    },
    "pt": {
        "name": "Portuguese",
        "tests": [
            "O Dr. Silva leu o artigo principal às 15h45 sobre pesca em 1999.",
            "Como se pronuncia 'entrepreneur'? O endereço custa R$99,99 mais 15% de impostos.",
            "O rato roeu a roupa do rei de Roma. Kkkk!",
            "O preço é de R$1.234,56 com desconto de 20%.",
            "Vamos nos encontrar às 19h30 na estação.",
        ]
    },
    "ru": {
        "name": "Russian",
        "tests": [
            "Доктор Иванов прочитал главную статью в 15:45 о рыбалке в 1999 году.",
            "Как произносится 'Worcestershire'? Адрес стоит 99,99€ плюс 15% НДС.",
            "Карл у Клары украл кораллы, а Клара у Карла украла кларнет. Лол!",
            "Цена составляет 1.234,56€ со скидкой 20%.",
            "Встретимся в 19:30 на вокзале.",
        ]
    },
    "ar": {
        "name": "Arabic",
        "tests": [
            "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات.",
            "السعر هو 99.99 دولار بالإضافة إلى 15% ضريبة.",
            "في عام 1999، كانت التكنولوجيا مختلفة تماماً.",
        ]
    },
    "hi": {
        "name": "Hindi",
        "tests": [
            "डॉ. शर्मा ने 1999 में मछली पकड़ने के बारे में मुख्य लेख दोपहर 3:45 बजे पढ़ा।",
            "'Entrepreneur' का उच्चारण कैसे करें? कीमत ₹99.99 है और 15% कर जोड़ें।",
            "कच्चा पापड़ पक्का पापड़। हाहा!",
        ]
    },
    "tr": {
        "name": "Turkish",
        "tests": [
            "Dr. Yılmaz, 1999 yılında balık tutma hakkındaki ana makaleyi 15:45'te okudu.",
            "Fiyat 99,99€ artı %15 KDV. www.site.com.tr adresinde.",
            "2024 yılında teknoloji çok gelişti.",
        ]
    },
    "pl": {
        "name": "Polish",
        "tests": [
            "Dr. Kowalski przeczytał główny artykuł o wędkarstwie w 1999 roku o 15:45.",
            "Cena wynosi 99,99€ plus 15% VAT. www.strona.pl",
            "W Szczebrzeszynie chrząszcz brzmi w trzcinie.",
        ]
    },
    "nl": {
        "name": "Dutch",
        "tests": [
            "Dr. de Vries las het hoofdartikel om 15:45 over vissen in 1999.",
            "De prijs is €99,99 plus 15% BTW. www.website.nl",
            "De kat krabt de krullen van de trap.",
        ]
    },
    "sv": {
        "name": "Swedish",
        "tests": [
            "Dr. Andersson läste huvudartikeln kl 15:45 om fiske 1999.",
            "Priset är 99,99€ plus 15% moms. www.hemsida.se",
            "Sju sjösjuka sjömän sköttes av sju sköna sjuksköterskor.",
        ]
    },
    "da": {
        "name": "Danish",
        "tests": [
            "Dr. Nielsen læste hovedartiklen kl. 15:45 om fiskeri i 1999.",
            "Prisen er 99,99€ plus 15% moms. www.hjemmeside.dk",
            "Røde grød med fløde koster 50 kr.",
        ]
    },
    "no": {
        "name": "Norwegian",
        "tests": [
            "Dr. Hansen leste hovedartikkelen kl. 15:45 om fiske i 1999.",
            "Prisen er 99,99€ pluss 15% mva. www.nettsted.no",
            "I dag er det 15. januar 2024.",
        ]
    },
    "fi": {
        "name": "Finnish",
        "tests": [
            "Dr. Virtanen luki pääartikkelin klo 15:45 kalastuksesta vuonna 1999.",
            "Hinta on 99,99€ plus 15% ALV. www.sivusto.fi",
            "Kokko, kokoo koko kokko! Koko kokkoko?",
        ]
    },
    "el": {
        "name": "Greek",
        "tests": [
            "Ο Δρ. Παπαδόπουλος διάβασε το κύριο άρθρο στις 15:45 για το ψάρεμα το 1999.",
            "Η τιμή είναι 99,99€ συν 15% ΦΠΑ.",
            "Το 2024 είναι έτος τεχνολογίας.",
        ]
    },
    "he": {
        "name": "Hebrew",
        "tests": [
            "ד\"ר כהן קרא את המאמר הראשי ב-15:45 על דיג ב-1999.",
            "המחיר הוא 99.99 דולר פלוס 15% מע\"ם.",
            "בשנת 2024 הטכנולוגיה התקדמה מאוד.",
        ]
    },
    "ms": {
        "name": "Malay",
        "tests": [
            "Dr. Ahmad membaca artikel utama pada 15:45 tentang memancing pada 1999.",
            "Harga adalah RM99.99 ditambah 15% cukai.",
            "Pada tahun 2024, teknologi sangat maju.",
        ]
    },
    "sw": {
        "name": "Swahili",
        "tests": [
            "Daktari Mwangi alisoma makala kuu saa 15:45 kuhusu uvuvi mwaka 1999.",
            "Bei ni $99.99 pamoja na kodi ya 15%.",
            "Mwaka 2024, teknolojia imekua sana.",
        ]
    },
}


class MultilingualTester:
    """Comprehensive multilingual testing suite"""

    def __init__(self, output_dir: str = "tests/output"):
        """
        Initialize tester

        Args:
            output_dir: Directory for test outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("="*80)
        print("CHATTERBOX MULTILINGUAL STRESS TEST SUITE")
        print("="*80)

        # Initialize pipeline
        self.pipeline = ChatterBoxPipeline(device="cuda" if torch.cuda.is_available() else "cpu")

    def test_language(self, lang_code: str, test_data: dict, output_format: str = "wav") -> dict:
        """
        Test a single language with all test sentences

        Args:
            lang_code: Language code
            test_data: Test data dictionary
            output_format: Output format for audio files

        Returns:
            Dictionary with test results
        """
        lang_name = test_data["name"]
        tests = test_data["tests"]

        print(f"\n{'='*80}")
        print(f"Testing {lang_name} ({lang_code}) - {len(tests)} test sentences")
        print(f"{'='*80}")

        results = {
            "language": lang_name,
            "code": lang_code,
            "total_tests": len(tests),
            "passed": 0,
            "failed": 0,
            "total_time": 0,
            "avg_time": 0,
            "tests": []
        }

        for i, text in enumerate(tests, 1):
            print(f"\n[{i}/{len(tests)}] Testing: {text[:60]}...")

            try:
                start_time = time.time()

                # Generate audio
                audio, sample_rate = self.pipeline.generate(
                    text,
                    language=lang_code,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.8
                )

                # Save audio
                output_path = f"{self.output_dir}/{lang_code}_{i:02d}"
                output_file = self.pipeline.save_audio(
                    audio, sample_rate, output_path, output_format
                )

                gen_time = time.time() - start_time
                duration = len(audio) / sample_rate
                rtf = duration / gen_time

                print(f"  ✅ Success!")
                print(f"     Generation time: {gen_time:.3f}s")
                print(f"     Audio duration: {duration:.2f}s")
                print(f"     RTF: {rtf:.2f}x")
                print(f"     Output: {output_file}")

                results["passed"] += 1
                results["total_time"] += gen_time
                results["tests"].append({
                    "text": text,
                    "status": "PASS",
                    "time": gen_time,
                    "duration": duration,
                    "rtf": rtf,
                    "output": output_file
                })

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results["failed"] += 1
                results["tests"].append({
                    "text": text,
                    "status": "FAIL",
                    "error": str(e)
                })

        results["avg_time"] = results["total_time"] / len(tests) if tests else 0

        return results

    def test_all_languages(self, output_format: str = "wav", languages: list = None) -> dict:
        """
        Test all languages or specified subset

        Args:
            output_format: Output format for audio files
            languages: List of language codes to test (None = all)

        Returns:
            Dictionary with all test results
        """
        if languages is None:
            languages = list(MULTILINGUAL_STRESS_TESTS.keys())

        all_results = {
            "total_languages": len(languages),
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "languages": {}
        }

        start_time = time.time()

        for lang_code in languages:
            if lang_code not in MULTILINGUAL_STRESS_TESTS:
                print(f"⚠️  Skipping unknown language: {lang_code}")
                continue

            test_data = MULTILINGUAL_STRESS_TESTS[lang_code]
            results = self.test_language(lang_code, test_data, output_format)

            all_results["languages"][lang_code] = results
            all_results["total_tests"] += results["total_tests"]
            all_results["total_passed"] += results["passed"]
            all_results["total_failed"] += results["failed"]

        all_results["total_time"] = time.time() - start_time

        return all_results

    def print_summary(self, results: dict):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        print(f"\nTotal Languages Tested: {results['total_languages']}")
        print(f"Total Test Sentences: {results['total_tests']}")
        print(f"Passed: {results['total_passed']} ✅")
        print(f"Failed: {results['total_failed']} ❌")
        print(f"Success Rate: {results['total_passed']/results['total_tests']*100:.1f}%")
        print(f"Total Time: {results['total_time']:.1f}s")

        print(f"\n{'Language':<15} {'Tests':<8} {'Passed':<8} {'Failed':<8} {'Avg Time':<12} {'Status'}")
        print("-"*80)

        for lang_code, lang_results in results["languages"].items():
            status = "✅ PASS" if lang_results["failed"] == 0 else "❌ FAIL"
            print(f"{lang_results['language']:<15} "
                  f"{lang_results['total_tests']:<8} "
                  f"{lang_results['passed']:<8} "
                  f"{lang_results['failed']:<8} "
                  f"{lang_results['avg_time']:<12.3f} "
                  f"{status}")

        print("\n" + "="*80)
        print(f"Test outputs saved to: {os.path.abspath(self.output_dir)}")
        print("="*80)


def main():
    """Run multilingual stress tests"""
    import argparse

    parser = argparse.ArgumentParser(description="ChatterBox Multilingual Stress Tests")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3", "raw"],
                        help="Output format (default: wav)")
    parser.add_argument("--languages", type=str, nargs="+",
                        help="Specific languages to test (default: all)")
    parser.add_argument("--output", type=str, default="tests/output",
                        help="Output directory")

    args = parser.parse_args()

    # Run tests
    tester = MultilingualTester(output_dir=args.output)
    results = tester.test_all_languages(
        output_format=args.format,
        languages=args.languages
    )
    tester.print_summary(results)


if __name__ == "__main__":
    main()
