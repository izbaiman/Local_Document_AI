#!/usr/bin/env python3
"""
generate_sample_docs.py
=======================
Creates 12 sample documents (mix of PDF and TXT) in ./sample_docs/
so the pipeline can be tested immediately without real documents.

Documents generated:
    invoice_1.pdf, invoice_2.pdf, invoice_3.pdf
    resume_1.pdf,  resume_2.pdf,  resume_3.pdf
    utility_1.pdf, utility_2.pdf, utility_3.txt
    other_1.txt,   other_2.pdf,   other_3.txt
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "sample_docs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Try to create PDFs using reportlab, fall back to plain text ────────────────

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
    print("[WARN] reportlab not installed — generating .txt files instead of PDFs.")
    print("       Install with: pip install reportlab")


def make_pdf(filename: str, content: str) -> None:
    """Write content to a PDF file using reportlab."""
    path = OUTPUT_DIR / filename
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 11)
    y = 720
    for line in content.split("\n"):
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = 720
        c.drawString(50, y, line)
        y -= 16
    c.save()
    print(f"  Created: {path.name}")


def make_txt(filename: str, content: str) -> None:
    """Write content to a plain text file."""
    # Replace .pdf extension if no reportlab
    if filename.endswith(".pdf") and not _HAS_REPORTLAB:
        filename = filename.replace(".pdf", ".txt")
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  Created: {path.name}")


def make_file(filename: str, content: str) -> None:
    if filename.endswith(".pdf"):
        if _HAS_REPORTLAB:
            make_pdf(filename, content)
        else:
            make_txt(filename, content)
    else:
        make_txt(filename, content)


# ══════════════════════════════════════════════════════════════════════════════
# Document contents
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENTS = {
    # ── Invoices ──────────────────────────────────────────────────────────────
    "invoice_1.pdf": """\
ACME Corporation
123 Business Park, New York, NY 10001
Tel: +1-212-555-0100

INVOICE

Invoice Number: INV-2025-0042
Date: January 15, 2025
Due Date: February 14, 2025

Bill To:
Globex Industries
456 Commerce Street
Chicago, IL 60601

Description                  Qty   Unit Price    Amount
--------------------------------------------------------
Web Design Services            1    $1,500.00   $1,500.00
SEO Optimisation Package       1      $350.00     $350.00
Monthly Maintenance            3      $100.00     $300.00

                                     Subtotal:  $2,150.00
                                         Tax (8%):  $172.00
                               TOTAL AMOUNT:  $2,322.00

Payment Terms: Net 30
Please remit payment to: bank@acmecorp.com
""",

    "invoice_2.pdf": """\
TechSupply Ltd.
78 Innovation Drive, Austin, TX 73301

TAX INVOICE
Invoice No.: TS-8821
Invoice Date: 03/22/2025
Payment Due: 04/21/2025

Vendor: TechSupply Ltd.
Customer: Bright Future Schools

Item                        Units    Rate      Total
-----------------------------------------------------
Laptop Computers (HP)          10   $850.00   $8,500.00
Wireless Keyboards             10    $45.00     $450.00
Mouse (Logitech)               10    $25.00     $250.00

                           Sub-total:          $9,200.00
                           Discount (5%):       -$460.00
                           VAT (10%):            $874.00
                           Total Amount Due:   $9,614.00

Purchase Order Ref: PO-2025-0088
""",

    "invoice_3.pdf": """\
FREELANCE SERVICES INVOICE

From: Maria Santos Design Studio
Email: maria@santosdesign.com

To: Omega Retail Group
Attention: Procurement Department

Invoice #: MSD-105
Date: February 28, 2025
Due Date: March 30, 2025

Services Rendered:
- Brand Identity Design Package      $2,000.00
- Social Media Graphics (30 posts)   $1,200.00
- Print-Ready Brochure (8 pages)       $800.00

                           Subtotal:  $4,000.00
                            Tax (0%):      $0.00
                             TOTAL:   $4,000.00

Bank Transfer: First National Bank
Account: 00123456789
Routing: 021000021
""",

    # ── Resumes ───────────────────────────────────────────────────────────────
    "resume_1.pdf": """\
James R. Mitchell
james.mitchell@email.com | (312) 555-0198 | LinkedIn: linkedin.com/in/jrmitchell

PROFESSIONAL SUMMARY
Experienced software engineer with 7 years of experience in full-stack web development,
cloud infrastructure, and team leadership.

EXPERIENCE

Senior Software Engineer — Cloudbase Inc.                      2020 – Present
  • Led a team of 5 engineers building microservices on AWS
  • Reduced deployment time by 40% through CI/CD pipeline improvements

Software Engineer — DataStream Analytics                       2018 – 2020
  • Built ETL pipelines processing 10M records/day
  • Stack: Python, Apache Spark, PostgreSQL

Junior Developer — WebWorks Agency                             2017 – 2018
  • Developed client-facing websites using React and Node.js

EDUCATION
Bachelor of Science in Computer Science — University of Illinois, 2017
GPA: 3.8 / 4.0

SKILLS
Python, JavaScript, TypeScript, React, AWS, Docker, Kubernetes, PostgreSQL, Redis

CERTIFICATIONS
AWS Certified Solutions Architect (2022)
""",

    "resume_2.pdf": """\
CURRICULUM VITAE

Aisha Patel
aisha.patel@gmail.com
Phone: +44 7911 123456

OBJECTIVE
Seeking a marketing manager role where I can leverage 5 years of experience
in digital marketing, brand strategy and team leadership.

WORK EXPERIENCE

Digital Marketing Manager — BrightBrand UK              Jan 2021 – Present
  Managed £500k annual ad budget across Google and Meta platforms.
  Increased organic traffic by 120% within 18 months.

Marketing Executive — Pulse Media Group                  Jun 2019 – Dec 2020
  Executed email campaigns achieving 28% open rate.
  Coordinated influencer partnerships with 50+ creators.

EDUCATION
Master of Science, Marketing Analytics — University of Edinburgh   2019
Bachelor of Commerce — University of Manchester                    2017

SKILLS
Google Analytics, HubSpot, Salesforce CRM, SEO/SEM, A/B Testing, SQL

References available on request.
""",

    "resume_3.pdf": """\
DAVID CHEN
davidchen@promail.com | 555-867-5309

Data Scientist | Machine Learning Engineer

PROFILE
Data scientist with 4 years of experience building production ML pipelines,
NLP models, and recommendation systems.

EXPERIENCE

Data Scientist II — NovaMind AI                           2022 – Present
  Built transformer-based NLP classifier (F1: 0.94) deployed in production.
  Mentored 2 junior data scientists.

Data Scientist — RetailIQ Corp                            2021 – 2022
  Designed recommendation engine boosting click-through by 18%.

Machine Learning Intern — FutureTech Labs                2020 – 2021
  Contributed to computer vision object detection project.

EDUCATION
M.S. Data Science — Stanford University                   2020
B.S. Statistics — UC Berkeley                             2018
GPA: 3.9 / 4.0

TECHNICAL SKILLS
Python, TensorFlow, PyTorch, Scikit-learn, SQL, Spark, Docker, GCP

PROJECTS
  GitHub: github.com/dchen-ds
""",

    # ── Utility Bills ─────────────────────────────────────────────────────────
    "utility_1.pdf": """\
METROPOLITAN ELECTRIC COMPANY
Customer Service: 1-800-555-POWER
www.metroelectric.com

ELECTRICITY BILL

Account Number: MEL-0045892-01
Customer Name: Sarah Thompson
Service Address: 88 Maple Street, Portland, OR 97201

Billing Period: January 1, 2025 – January 31, 2025
Statement Date: February 3, 2025
Payment Due Date: February 20, 2025

METER READING SUMMARY
Previous Reading (Dec 31): 43,210 kWh
Current Reading  (Jan 31): 43,892 kWh
Usage This Period:             682 kWh

CHARGES
Energy Charge (682 kWh × $0.12):     $81.84
Distribution Charge:                  $12.50
Base Service Charge:                   $8.00
Taxes & Fees:                          $5.22
                        Amount Due:  $107.56

Previous Balance:      $0.00
                TOTAL DUE: $107.56

Please pay by February 20, 2025 to avoid late fees.
""",

    "utility_2.pdf": """\
SUNSTATE WATER & SEWER AUTHORITY
PO Box 4400, Phoenix, AZ 85001
(602) 555-7890

WATER BILL

Account #: SWS-2291047
Service Period: December 2024
Statement Date: January 5, 2025
Due Date: January 25, 2025

Billing Name: Carlos Rivera
Property Address: 22 Desert Rose Lane, Scottsdale, AZ 85251

USAGE SUMMARY
Meter Number: 0027-AZ
Previous Reading: 112,480 gallons
Current Reading:  113,960 gallons
Consumption:        1,480 gallons
Usage kWh equivalent: 87.3 kWh (pumping energy estimate)

CURRENT CHARGES
Water Service (1,480 gal):       $23.40
Sewer Service:                   $18.75
Stormwater Charge:                $4.00
State Assessment:                 $1.85
                  Current Charges: $48.00
Previous Balance:                  $0.00
                  Amount Due: $48.00
""",

    "utility_3.txt": """\
LAKEVIEW GAS COMPANY
123 Energy Way, Chicago, IL 60602
Helpline: 1-888-555-LGAS

NATURAL GAS BILL

Account Number: LGC-887-4421-9
Customer: Jennifer Williams
Service Address: 5501 North Oak Ave, Chicago, IL 60640

Statement Date: January 10, 2025
Billing Period: Dec 15, 2024 – Jan 12, 2025
Payment Due: January 30, 2025

USAGE DETAILS
Therms Used This Period:        85.4 therms
kWh Equivalent:                2,501 kWh
Average Daily Usage:             2.9 therms/day

CHARGE BREAKDOWN
Gas Supply (85.4 therms @ $0.72):   $61.49
Distribution Charge:                 $14.20
Customer Service Charge:              $6.00
Efficiency Surcharge:                 $1.80
Taxes (City + State):                 $5.23
                       AMOUNT DUE:   $88.72

Remit payment to: Lakeview Gas Company, PO Box 4400, Chicago IL 60602
""",

    # ── Other / Unclassifiable ────────────────────────────────────────────────
    "other_1.txt": """\
BOARD MEETING MINUTES
Company: Horizon Ventures LLC
Date: March 5, 2025
Location: 300 Corporate Blvd, Suite 400, Boston, MA

Attendees:
  - Robert Kim (CEO)
  - Priya Sharma (CFO)
  - Tom Walsh (CTO)
  - Legal Counsel: Nina Okafor

Agenda Item 1: Q4 2024 Financial Review
  CFO presented Q4 results. Revenue increased 22% YoY to $4.2M.
  Operating expenses rose 11% due to new hires.

Agenda Item 2: Product Roadmap for 2025
  CTO outlined three major initiatives:
  (a) Mobile application launch — Q2 2025
  (b) API partner programme — Q3 2025
  (c) Enterprise tier rollout — Q4 2025

Agenda Item 3: Series B Fundraising Update
  CEO reported term sheets received from two VCs.
  Target close: April 2025.

Next meeting: April 2, 2025 at 10:00 AM EST.
Minutes recorded by: Executive Assistant
""",

    "other_2.pdf": """\
REAL ESTATE PURCHASE AGREEMENT

This Agreement is entered into on February 14, 2025 by and between:

Seller: Robert & Lisa Donovan
Buyer:  Kevin Park

Property Address: 1204 Elmwood Drive, Nashville, TN 37201

Purchase Price: $385,000.00
Earnest Money Deposit: $10,000.00 (due within 5 days)
Closing Date: March 28, 2025

Terms and Conditions:
1. The property is sold "as-is" subject to inspection contingency.
2. Financing contingency: Buyer to obtain mortgage approval within 21 days.
3. Home inspection to be completed by February 25, 2025.
4. Seller will provide clear title at closing.
5. Property taxes prorated to closing date.

HOA Fees: $220/month
Current Mortgage Balance (Seller): $210,000.00

Both parties agree to the above terms.

Seller Signature: ___________________ Date: __________
Buyer Signature:  ___________________ Date: __________
""",

    "other_3.txt": """\
ACADEMIC RESEARCH PAPER ABSTRACT

Title: Deep Learning Approaches for Satellite Image Segmentation in
       Urban Planning Applications

Authors: Dr. Emma Larsson, Prof. Kwame Asante, Dr. Yuki Tanaka
Institution: Institute for Geospatial Research, University of Amsterdam
Published: Urban Computing Journal, Vol. 18, Issue 2, 2025

Abstract:
This paper presents a comparative study of convolutional neural network (CNN)
architectures for semantic segmentation of high-resolution satellite imagery
in dense urban environments. We evaluate U-Net, DeepLab v3+, and a novel
hybrid transformer-CNN architecture on a dataset of 2,400 labelled images
from 12 cities across four continents.

Results indicate that our hybrid architecture achieves a mean Intersection-
over-Union (mIoU) of 0.87, outperforming standard U-Net (0.81) and DeepLab
v3+ (0.83) on the urban segmentation task.

The proposed model demonstrates particular robustness to:
  • Seasonal variation in vegetation
  • Shadows from high-rise buildings
  • Mixed-use zoning patterns

Dataset and code are openly available at: github.com/igs-ams/urban-seg

Keywords: semantic segmentation, satellite imagery, CNN, urban planning,
          remote sensing, deep learning
""",
}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nGenerating {len(DOCUMENTS)} sample documents in {OUTPUT_DIR}/\n")
    for fname, content in DOCUMENTS.items():
        make_file(fname, content)
    print(f"\nDone! {len(DOCUMENTS)} files created.\n")
