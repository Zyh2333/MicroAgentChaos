#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import locust.stats
from locust import HttpUser, between, task

locust.stats.CSV_STATS_INTERVAL_SEC = 5  # default is paymentservice_timeout second
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 5  # Determines how often the data is flushed to disk, default is 10 seconds

from locust import HttpUser, task, between
import random
import os

# The service currently under fault injection, passed in by chaos_service.sh
# via an environment variable (default pdf-parsing). locust uses it to bias the
# request distribution toward tasks that can reach the faulty service, producing
# more call chains through it within the fault window.
FAULT_SERVICE = os.getenv("FAULT_SERVICE", "agent-network-pdf-parsing")

# Weight bases (larger = higher chance of being sampled)
W_FAULT_COMPLEX = 6   # hits the faulty service AND is a complex multi-hop chain
W_FAULT_SIMPLE = 3     # hits the faulty service AND is a simple task
W_OTHER_COMPLEX = 2    # does not hit the faulty service, complex task
W_OTHER_SIMPLE = 1     # does not hit the faulty service, simple task


def task_services(task_def):
    """Service call chain a task actually traverses.

    An explicitly declared `services` takes precedence; otherwise it falls back
    to the task's own `name` (for a single-service SIMPLE task this is the only
    service it calls).
    """
    return task_def.get("services", [task_def["name"]])


def task_weight(task_def):
    """Compute the sampling weight from whether the task hits the current faulty service and its complexity."""
    hits_fault = FAULT_SERVICE in task_services(task_def)
    is_complex = task_def.get("type") == "COMPLEX"
    if hits_fault and is_complex:
        return W_FAULT_COMPLEX
    if hits_fault:
        return W_FAULT_SIMPLE
    if is_complex:
        return W_OTHER_COMPLEX
    return W_OTHER_SIMPLE


class MDOCUser(HttpUser):
    wait_time = between(5, 10)

    # 1. Pool of publicly accessible file resources
    FILE_RESOURCES = {
        # PDF files - very short neutral PDFs (short body to reduce token usage).
        # Note 1: the longer neutral PDFs below are kept commented out; uncomment
        #         them when needed.
        # Note 2: the original government-gazette PDF is still kept on the server,
        #         registered as fault_injection.py's "content_compliance"
        #         (violates laws/regulations) fault case; do not delete it.
        "pdf": [
            # Short doc: PDF test file (~57 words, orimi.com pdf-test)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c2hvcnRfcGRmdGVzdC5wZGY&user=exp",

            # Short doc: PDF sample (~173 words, unec.edu.az pdf-sample)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c2hvcnRfc2FtcGxlLnBkZg&user=exp",

            # Short doc: XSL-FO basic-link sample (~145 words, antennahouse)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c2hvcnRfbGluay5wZGY&user=exp",

            # ---- Longer neutral PDFs below, commented out for now; uncomment when needed ----
            # Attention Is All You Need (arXiv:1706.03762)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=YXR0ZW50aW9uX2lzX2FsbF95b3VfbmVlZC5wZGY&user=exp",
            # BERT pretraining paper (arXiv:1810.04805)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=YmVydF9wcmV0cmFpbmluZ19wYXBlci5wZGY&user=exp",
            # Somatosensory neuroscience textbook sample (css4.pub somatosensory)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c29tYXRvc2Vuc29yeV90ZXh0Ym9vay5wZGY&user=exp",
            # USENIX conference paper sample (css4.pub usenix example)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=dXNlbml4X2V4YW1wbGVfcGFwZXIucGRm&user=exp",
            # Drylab science newsletter sample (css4.pub drylab newsletter)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZHJ5bGFiX25ld3NsZXR0ZXIucGRm&user=exp",
        ],

        # Excel files - public data from mainland China
        "excel": [
            # SAMR: pastry spot-check failed-product info (orig: samr.gov.cn 090ad9e0...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMC54bHN4&user=exp",

            # SAMR: vegetable-product spot-check failed-product info (orig: samr.gov.cn 0595a450...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMS54bHN4&user=exp",

            # SAMR: fruit-product spot-check failed-product info (orig: samr.gov.cn f8ceb0d7...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMi54bHN4&user=exp",

            # SAMR: instant-food spot-check failed-product info (orig: samr.gov.cn bc79e9e8...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMy54bHN4&user=exp",

            # Central University of Finance and Economics: remote student-record archive lookup form (orig: archives.cufe.edu.cn ...2F76.xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfNC54bHN4&user=exp",
        ],

        # Word documents - public documents from mainland China
        "word": [
            # Shenzhen Longgang ENT Hospital: registration form
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=MTIzNTE5MzUuZG9jeA&user=exp",
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=5Lit5Zu96ZO26KGMMjAyNuW5tOagoeWbreWcsOWbveWutuWKqeWtpui0t-asvuWtpueUn-WcqOe6v-eUs-ivt-aMh-WNly5kb2N4&user=exp",
            # SAMR: tips on some failed inspection items
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=6ZmE5Lu2MSvpg6jliIbkuI3lkIjmoLzmo4Dpqozpobnnm67lsI_nn6Xor4YuZG9jeA&user=exp",
        ],

        # Images - simple images / OCR test
        "image_ocr": [
            # State Council: administrative map of the Macao SAR (orig: https://www.gov.cn/zhengce/aomen.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzAuanBlZw&user=exp",

            # NATCM planning-related image (orig: ...W020230506781423064462.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzEuanBlZw&user=exp",

            # NATCM planning-related image (orig: ...W020230506781423164720.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzIuanBlZw&user=exp",

            # NATCM planning-related image (orig: ...W020230506781423260634.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzMuanBlZw&user=exp",

            # NATCM planning-related image (orig: ...W020230506781423338851.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzQuanBlZw&user=exp",

            # # Environment and agent perception architecture diagram
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=Q2hhdEdQVCBJbWFnZSAyMDI25bm0N-aciDE15pelIDE3XzM0XzQwLnBuZw&user=exp"
        ],

        # Complex images - charts / PNG / JPG etc.
        "image_complex": [
            # NATCM planning-related PNG (orig: ...W020230506781422832145.png)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8wLnBuZw&user=exp",

            # NATCM planning-related JPG (orig: ...W020230506781422944424.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8xLmpwZWc&user=exp",

            # NATCM planning-related JPG (orig: ...W020230506781423064462.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8yLmpwZWc&user=exp",

            # NATCM planning-related JPG (orig: ...W020230506781423164720.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8zLmpwZWc&user=exp",

            # NATCM planning-related JPG (orig: ...W020230506781423260634.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF80LmpwZWc&user=exp",
            # Landscape painting
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=T0RGRk1rWkNOemRCTjBJd05FRXlPRGxHTlRjd056STBPVGcxTWpoQk5VWGt1SzNsbTczbHNiSG1zTFRubEx2cHU0VGxzYkhrdXBIbXRiZm1sNlhsaDdveU1ESTJNRGt3Tmkwd09ERTJOVGd1Y0c1bi5wbmc&user=exp",
            # Microservice architecture diagram
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=Q2hhdEdQVCBJbWFnZSAyMDI25bm0N-aciDE15pelIDAwXzQzXzQwLnBuZw&user=exp",
        ],
    }

    # 2. Task configuration list
    TASK_DEFINITIONS = [
        # ==================== Simple tasks ====================
        # CSV generation
        {
            "name": "agent-network-csv-gen",
            "type": "SIMPLE",
            "prompt": "Generate employee data in CSV format containing name, age, department, and salary, 20 records in total."
        },
        {
            "name": "agent-network-csv-gen",
            "type": "SIMPLE",
            "prompt": "Generate a student grade table in CSV containing student ID, name, Chinese, math, and English scores, 30 records in total."
        },
        {
            "name": "agent-network-csv-gen",
            "type": "SIMPLE",
            "prompt": "Generate product sales data in CSV containing product ID, name, category, unit price, and sales volume, 50 records in total."
        },

        # Route planning
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "Plan a driving route from Beijing Capital Airport to the city center, avoiding traffic congestion."
        },
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "Plan a public transit route from Shanghai Hongqiao Railway Station to the Bund."
        },
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "Plan a subway route from Canton Tower to Baiyun Airport, including transfer information."
        },
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "Plan the fastest route within Shenzhen from Nanshan Science Park to Bao'an Airport."
        },

        # Excel generation
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "Generate a sales forecast Excel spreadsheet for Q1 2024, containing month, product category, estimated sales, and actual sales."
        },
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "Generate an employee attendance statistics Excel sheet, containing name, department, days attended, days of leave, and overtime hours."
        },
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "Generate a household monthly budget Excel sheet, containing income, expense categories, budgeted amount, and actual amount."
        },
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "Generate an inventory management Excel sheet, containing product code, name, stock quantity, safety stock, and supplier."
        },

        # Image generation
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "Generate a high-definition image of 'a cyberpunk-style futuristic city at night with neon lights'."
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "Generate an image of 'Chinese landscape painting style, Huangshan sea of clouds, sunrise'."
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "Generate an interior design image of 'a modern minimalist living room design with white as the main color'."
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "Generate an illustration of 'a cute cartoon kitten in watercolor style'."
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "Generate a concept image of 'a high-tech AI robot, blue tones, futuristic feel'."
        },

        # Travel planning
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "Create a detailed 'Dali, Yunnan 3-day 2-night' travel guide, including accommodation suggestions, attraction recommendations, and food recommendations."
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "Create a 'Chengdu 5-day 4-night' family trip itinerary, including the Giant Panda Base, Dujiangyan and other attractions."
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "Create a "Xi'an 3-day history and culture tour" plan, including the Terracotta Army, Huaqing Pool, the ancient city wall, etc."
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "Create a 'Hangzhou West Lake 2-day tour' guide, including Lingyin Temple, Longjing Village and other attractions."
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "Create a 'Sanya seaside vacation 4-day 3-night' itinerary, including beaches, seafood, and water activities."
        # },

        # Text summarization
        # {
        #     "name": "agent-network-summarizer",
        #     "type": "SIMPLE",
        #     "prompt": "Summarize the core points of the following text: 'Artificial intelligence is changing the world, and large-model technology makes natural language processing more efficient. Deep learning algorithms have made breakthroughs in image recognition, speech recognition and other fields.'"
        # },
        # {
        #     "name": "agent-network-summarizer",
        #     "type": "SIMPLE",
        #     "prompt": "Summarize the main content of this news report: 'In 2024 China's economy grew steadily, with GDP up 5.2% year on year, the consumer market continuing to recover, and high-tech industries developing rapidly.'"
        # },
        # {
        #     "name": "agent-network-summarizer",
        #     "type": "SIMPLE",
        #     "prompt": "Extract the key decision points from the following meeting minutes: 'The meeting discussed the new product launch plan, marketing strategy, staffing adjustments and other topics.'"
        # },

        # Word document generation
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "Generate a Word document outline for a 'project proposal', containing project background, objectives, implementation plan, and budget."
        },
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "Generate a Word document template for an 'annual work summary report', containing work review, achievements, problem analysis, and next year's plan."
        },
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "Generate a Word document structure for a 'business plan', containing market analysis, business model, and financial projections."
        },
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "Generate a Word document for an 'employee training plan', containing training objectives, course content, schedule, and assessment methods."
        },

        # ==================== File parsing tasks ====================
        # PDF parsing
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "Please parse this PDF file and extract the document title, main sections, and key information: {url}"
        },
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "Analyze the content structure of this PDF file, list all tables, and extract the data: {url}"
        },
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "Extract all dates, numbers, and key metrics from this PDF file: {url}"
        },
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "Parse the PDF file and summarize the core content and main conclusions of the document: {url}"
        },

        # Excel parsing
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "Read this Excel file and list all worksheet names and the column names of each sheet: {url}"
        },
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "Analyze this Excel file and compute the sum, average, maximum, and minimum of the numeric columns: {url}"
        },
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "Extract the first 10 rows of data from this Excel file and convert them to JSON format: {url}"
        },
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "Analyze the data trends in the Excel file and identify outliers and patterns: {url}"
        },

        # Word parsing
        {
            "name": "agent-network-word-parsing",
            "type": "SIMPLE",
            "requires_file": "word",
            "prompt": "Parse this Word document and extract all headings and paragraph content: {url}"
        },
        {
            "name": "agent-network-word-parsing",
            "type": "SIMPLE",
            "requires_file": "word",
            "prompt": "Extract all table data from this Word document: {url}"
        },
        {
            "name": "agent-network-word-parsing",
            "type": "SIMPLE",
            "requires_file": "word",
            "prompt": "Analyze the structure of the Word document and extract the table of contents and section information: {url}"
        },

        # OCR recognition
        {
            "name": "agent-network-ocr",
            "type": "SIMPLE",
            "requires_file": "image_ocr",
            "prompt": "Recognize all text content in this image, including Chinese and English: {url}"
        },
        {
            "name": "agent-network-ocr",
            "type": "SIMPLE",
            "requires_file": "image_ocr",
            "prompt": "Extract the numbers, dates, and key information from the image: {url}"
        },
        {
            "name": "agent-network-ocr",
            "type": "SIMPLE",
            "requires_file": "image_ocr",
            "prompt": "Recognize the text in the image and convert it to editable text format: {url}"
        },

        # Image analysis
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "Describe the content of this image in detail, including colors, objects, scene, and layout: {url}"
        },
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "Analyze the style, theme, and visual elements of this image: {url}"
        },
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "Identify the brand logos, text, and main elements in the image: {url}"
        },
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "Evaluate the quality, composition, and color scheme of this image: {url}"
        },

        # ==================== Combined / long-chain tasks ====================
        # Image + report generation
        {
            "name": "agent-network-combo-image-report",
            "type": "COMPLEX",
            "services": ["agent-network-image", "agent-network-word-gen"],
            "requires_file": "image_complex",
            "prompt": "First analyze the content of this image ({url}), then generate a detailed 'visual analysis report' based on the image content, including scene description, element analysis, color analysis, and recommendations."
        },
        {
            "name": "agent-network-combo-ocr-summary",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "First recognize the text in this image ({url}), then summarize and categorize the recognized content, and generate a well-structured PDF summary report."
        },
        {
            "name": "agent-network-combo-ocr-pdf-transcript",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "Step 1: Perform OCR on the image ({url}) and fully extract the Chinese and English text, numbers, and dates in it; "
                      "Step 2: Correct obvious recognition errors while preserving the original paragraph hierarchy; "
                      "Step 3: Generate an archivable PDF text transcription document from the organized content."
        },
        {
            "name": "agent-network-combo-ocr-pdf-key-info",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "Step 1: Recognize all text in the image ({url}); "
                      "Step 2: Extract important information such as titles, dates, locations, organization names, and key numbers; "
                      "Step 3: Generate a PDF report containing excerpts of the original text and a table of key information."
        },
        {
            "name": "agent-network-combo-ocr-pdf-bilingual",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "First perform OCR on the image ({url}), recognizing the Chinese and English content separately; "
                      "then organize the recognition results by language, and generate a PDF document containing the recognized original text, a content overview, and the categorized results."
        },
        {
            "name": "agent-network-combo-ocr-pdf-audit",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "Perform OCR on the image ({url}), checking item by item and flagging suspected recognition errors, blurry characters, and missing content, "
                      "and finally generate a PDF quality-inspection report containing the recognition results, an issue list, and confidence notes."
        },

        # Excel + data analysis
        {
            "name": "agent-network-combo-excel-analysis",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-word-gen"],
            "requires_file": "excel",
            "prompt": "Read this Excel file ({url}), analyze the data trends, and generate a 'data analysis report' containing key findings, chart suggestions, and improvement recommendations."
        },
        {
            "name": "agent-network-combo-excel-visualization",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-image-gen"],
            "requires_file": "excel",
            "prompt": "Analyze the data in this Excel file ({url}), then generate a 'business insight report' that includes data visualizations."
        },

        # PDF + summary
        {
            "name": "agent-network-combo-pdf-report",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "Parse this PDF file ({url}), extract key information, then generate a 'document summary report' containing background, main content, conclusions, and recommendations."
        },
        {
            "name": "agent-network-combo-pdf-excel",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-gen"],
            "requires_file": "pdf",
            "prompt": "Extract all table data from the PDF file ({url}) and organize it into a structured Excel-format description."
        },

        # Multi-step planning tasks
        {
            "name": "agent-network-combo-travel-budget",
            "type": "COMPLEX",
            "services": ["agent-network-planner", "agent-network-excel-gen"],
            "prompt": "I need to travel to Ili, Xinjiang for 7 days. First generate a detailed itinerary plan, then generate a 'budget estimate sheet' based on the itinerary, covering costs such as transportation, accommodation, dining, and tickets."
        },
        {
            "name": "agent-network-combo-marketing-plan",
            "type": "COMPLEX",
            "services": ["agent-network-planner", "agent-network-word-gen"],
            "prompt": "Develop a marketing plan for a new smartphone. First conduct a market analysis, then generate a 'marketing strategy document' containing target users, channel strategy, and promotion plan."
        },
        {
            "name": "agent-network-combo-product-launch",
            "type": "COMPLEX",
            "services": ["agent-network-planner", "agent-network-word-gen"],
            "prompt": "Plan a new product launch event. First draw up the event schedule, then generate an 'execution plan' containing the timeline, staff assignments, and materials list."
        },
        {
            "name": "agent-network-combo-data-visualization",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-image-gen"],
            "prompt": "I have a set of sales data. Please analyze the characteristics of the data, then generate a 'data visualization plan' that recommends suitable chart types and presentation methods."
        },

        # Document conversion
        {
            "name": "agent-network-combo-pdf-to-word",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "Parse the PDF file ({url}), extract the content, and reorganize it into Word document format, including structured elements such as headings, paragraphs, and lists."
        },
        {
            "name": "agent-network-combo-word-to-ppt",
            "type": "COMPLEX",
            "services": ["agent-network-word-parsing", "agent-network-word-gen"],
            "requires_file": "word",
            "prompt": "Analyze the content of the Word document ({url}), then generate a PPT outline containing the title, key points, and visual suggestions for each slide."
        },

        # ==================== 3+ service long-chain COMPLEX tasks ====================
        # These tasks explicitly plan 3 or more service call chains and mostly pass
        # through the pdf-parsing faulty service, producing richer multi-hop call
        # chains within the fault window to observe fault propagation along them.
        {
            "name": "agent-network-combo-pdf-excel-word",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-gen", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "Step 1: Parse this PDF file ({url}) and extract the tables and key data in it; "
                      "Step 2: Organize the extracted tables into structured Excel data; "
                      "Step 3: Based on the Excel data, generate a complete Word analysis report containing a data overview, trend analysis, and conclusions and recommendations."
        },
        {
            "name": "agent-network-combo-pdf-excel-chart",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-gen", "agent-network-image-gen"],
            "requires_file": "pdf",
            "prompt": "Step 1: Parse this PDF file ({url}) and extract the tables and numeric metrics in it; "
                      "Step 2: Organize the extracted data into a structured Excel spreadsheet; "
                      "Step 3: Based on this Excel data, generate a data visualization chart that intuitively presents the key metrics."
        },
        {
            "name": "agent-network-combo-pdf-excel-analysis-report",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-parsing",
                         "agent-network-excel-gen", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "Step 1: Parse the PDF file ({url}) and extract all tables in it; "
                      "Step 2: Read the tables in as Excel data and perform statistical analysis (sum / average / outliers); "
                      "Step 3: Reorganize the analysis results into a new Excel summary table; "
                      "Step 4: Generate a Word-format 'Data Insight Report' explaining the key findings and improvement recommendations."
        },
        {
            "name": "agent-network-combo-ocr-excel-report",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-excel-gen", "agent-network-word-gen"],
            "requires_file": "image_ocr",
            "prompt": "Step 1: Perform OCR on the image at this explicit URL ({url}) and extract the text, numbers, and dates in it; "
                      "Step 2: Organize the OCR results into a structured Excel spreadsheet by information type; "
                      "Step 3: Based on the recognition and organization results, generate a Word-format 'Image Text Recognition Report' containing text excerpts, key information, and a categorized summary."
        },
        {
            "name": "agent-network-combo-image-word",
            "type": "COMPLEX",
            "services": ["agent-network-image", "agent-network-word-gen"],
            "requires_file": "image_complex",
            "prompt": "Step 1: Analyze the scene, layout, colors, and main visual elements of the image ({url}); "
                      "Step 2: Combine the visual analysis and recognition results to generate a structured Word image analysis report."
        },
        {
            "name": "agent-network-combo-ocr-excel-chart",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-excel-gen", "agent-network-image-gen"],
            "requires_file": "image_ocr",
            "prompt": "Step 1: Recognize the text, numbers, dates, and table information in the image ({url}); "
                      "Step 2: Organize the quantifiable information into a structured Excel spreadsheet; "
                      "Step 3: Based on the key data in the table, generate a visualization chart."
        },
        {
            "name": "agent-network-combo-pdf-word-pdf",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-word-gen", "agent-network-pdf-gen"],
            "requires_file": "pdf",
            "prompt": "Step 1: Parse the PDF file ({url}) and extract the title, sections, and body text; "
                      "Step 2: Reorganize the extracted content into a well-structured Word document; "
                      "Step 3: Generate a new, clean-layout PDF file from the organized document."
        },
        {
            "name": "agent-network-combo-excel-chart-pdf",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-image-gen", "agent-network-pdf-gen"],
            "requires_file": "excel",
            "prompt": "Step 1: Parse the Excel file ({url}) and analyze the key metrics and data trends; "
                      "Step 2: Generate appropriate data visualization charts based on the analysis results; "
                      "Step 3: Lay out the metric explanations and charts into a PDF data report."
        },
        {
            "name": "agent-network-combo-word-image-pdf",
            "type": "COMPLEX",
            "services": ["agent-network-word-parsing", "agent-network-image-gen", "agent-network-pdf-gen"],
            "requires_file": "word",
            "prompt": "Step 1: Parse the Word document ({url}) and extract the title, sections, key content, and data; "
                      "Step 2: Generate a matching infographic based on the document's key points; "
                      "Step 3: Integrate the original key points and the infographic into a PDF briefing."
        },
    ]

    # Sampling weights aligned one-to-one with TASK_DEFINITIONS: tasks hitting
    # the current faulty service (FAULT_SERVICE) get higher weight, and complex
    # multi-hop chains rank above simple tasks.
    TASK_WEIGHTS = [task_weight(td) for td in TASK_DEFINITIONS]

    @task
    def run_mixed_tasks(self):
        # Randomly pick a task definition by fault-aware weight (biased toward the faulty service)
        task_def = random.choices(self.TASK_DEFINITIONS, weights=self.TASK_WEIGHTS, k=1)[0]

        # Resolve the file URL (if needed)
        final_prompt = task_def["prompt"]
        if task_def.get("requires_file"):
            file_type = task_def["requires_file"]
            file_urls = self.FILE_RESOURCES.get(file_type, [])
            if file_urls:
                # Randomly pick a file URL
                file_url = random.choice(file_urls)
                final_prompt = final_prompt.format(url=file_url)
            else:
                print(f"Warning: No URL found for file type {file_type}")

        # Build the request payload
        payload = {
            "flowId": "@cn.com.thingo.intelligentAgentPlatform.taskScheduling/FLOW_OPEN_TASK",
            "params": {
                "openTask": {
                    "userId": "CDA7B6E29E2B4CBBB53E424A68A1B85C",
                    "organizeId": "C122FBF575E94BF8867580A535E20BD3",
                },
                "task": final_prompt
            }
        }

        # Send the request and tag the task type
        task_name = task_def["name"]
        task_type = task_def["type"]

        import uuid
        id = uuid.uuid4()
        # One line per request to correlate id/type/name/prompt while reducing log noise
        print(f"id:{id} {task_type}/{task_name} | {final_prompt}")

        with self.client.post(
                "/api/engine/flow",
                json=payload,
                name=f"{task_type}/{task_name}",
                catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code} for {task_name}")
