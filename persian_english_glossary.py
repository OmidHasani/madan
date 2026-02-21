# -*- coding: utf-8 -*-
"""
واژه‌نامهٔ فارسی–انگلیسی برای اصطلاحات فنی مستندات (Shop Manual).
برای هر سوالی که متن فارسی دارد، عبارت‌های انگلیسی این واژه‌نامه به کوئری جستجو اضافه می‌شود
تا بازیابی به چانک‌های درست مستندات انگلیسی برسد.

هر آیتم: (لیست عبارت‌های فارسی/ترکیبی که در سوال جستجو می‌شوند, رشتهٔ کلیدواژهٔ انگلیسی برای جستجو)
"""

# نوع: لیست تاپل‌های (list[str], str)
# اگر هر یک از رشته‌های لیست اول در سوال کاربر باشد، کلیدواژه‌های انگلیسی (رشتهٔ دوم) به کوئری اضافه می‌شود.
from typing import List, Tuple

PERSIAN_ENGLISH_GLOSSARY: List[Tuple[List[str], str]] = [
    # موتور چرخشی / سوئینگ
    (
        ["موتور چرخشی", "موتور سوئینگ", "موتور چرخش", "سوئینگ موتور"],
        "swing motor swing circle swing machinery structure function maintenance standard "
        "reduction ratio 8.615 35.716 inner race 112 teeth ring gear grease 65 G2-LI "
        "clearance bearing axial standard repair limit replace PC800 800LC-8 soft zone S position plug "
    ),
    # سیلندر بوم
    (
        ["سیلندر بوم", "بوم سیلندر", "جک بوم", "بوم"],
        "boom cylinder structure function maintenance standard stroke rod head bottom "
        "PC800 800LC-8 hydraulic "
    ),
    # سیلندر بازو
    (
        ["سیلندر بازو", "بازو سیلندر", "جک بازو", "بازو"],
        "arm cylinder structure function maintenance standard stroke rod head bottom "
        "PC800 800LC-8 hydraulic "
    ),
    # سیلندر باکت
    (
        ["سیلندر باکت", "باکت سیلندر", "جک باکت", "باکت"],
        "bucket cylinder structure function maintenance standard stroke rod head bottom "
        "curl dump PC800 800LC-8 hydraulic "
    ),
    # موتور حرکت / تراول
    (
        ["موتور حرکت", "موتور تراول", "تراول موتور", "موتور چرخ زنجیر"],
        "travel motor final drive structure function maintenance standard "
        "PC800 800LC-8 sprocket track "
    ),
    # نشت روغن / اندازه‌گیری نشت
    (
        ["نشت روغن", "اندازه‌گیری نشت", "نشت سیلندر", "نشت موتور"],
        "measuring oil leakage measuring leakage from boom cylinder arm cylinder bucket cylinder "
        "swing motor travel motor center swivel joint measuring device flange plug inspection port "
    ),
    # تخلیه هوا
    (
        ["تخلیه هوا", "هوا از هر بخش", "تخلیه هوا از هر بخش", "هواگیری"],
        "Bleeding air from each part Air bleeding item Contents of work "
        "Bleeding air from hydraulic pump Bleeding air from cylinder Bleeding air from swing motor "
        "Bleeding air from travel motor Checking oil level and starting work "
        "bleeder work equipment pump fan pump stroke end piston rod arm cylinder bucket cylinder "
        "parking brake circuit safety valve circuit travel motor cover sight gauge "
    ),
    # دیود
    (
        ["دیود", "بررسی دیود", "بازرسی دیود"],
        "diode inspection procedures assembled-type diode single diode circuit tester "
        "anode cathode continuity resistance range digital analog test lead needle swing "
    ),
    # پمپ و هیدرولیک
    (
        ["پمپ هیدرولیک", "پمپ هیدرولیکی", "هیدرولیک پمپ", "پمپ اصلی"],
        "hydraulic pump HPV375 front pump rear pump structure function maintenance standard "
        "PC800 800LC-8 "
    ),
    (
        ["پمپ فن", "فن پمپ", "پمپ خنک کننده"],
        "fan pump cooling fan pump LPV90 structure function maintenance standard "
    ),
    # شیرها و سوپاپ
    (
        ["شیر کنترل", "شیر کنترول", "کنترل ولو", "والو کنترل"],
        "control valve 5-spool 4-spool L.H. R.H. main valve relief valve "
        "PC800 800LC-8 structure function maintenance standard "
    ),
    (
        ["شیر اطمینان", "رلیف", "رلیف والو"],
        "relief valve safety valve main relief pressure setting MPa kg/cm2 "
    ),
    (
        ["شیر برقی", "سولنوئید", "سولنوئید والو"],
        "solenoid valve swing brake travel speed machine push-up heavy lift straight travel "
    ),
    # شیر بای‌پس (موتور / EGR) — همهٔ کدهای خطای مرتبط تا بازیابی عیب‌یابی درست شود
    (
        ["شیر بای پس", "شیر بایپس", "بای پس", "بایپس", "مشکل شیر بای پس", "مشکل بایپس"],
        "bypass valve BP valve EGR bypass valve solenoid lift sensor "
        "Failure code CA1626 CA1627 CA1628 CA1629 CA1631 CA1632 "
        "BP Valve Sol Current High Low Bypass Valve Servo Error BP Valve Pos Sens "
        "troubleshooting Possible causes standard value in normal state "
    ),
    # PPC و اپراتور
    (
        ["شیر PPC", "PPC", "شیر پی پی سی", "اپراتور"],
        "PPC valve pilot pressure control lever boom swing arm bucket travel "
    ),
    # گریس و روغن
    (
        ["گریس", "گریسکاری", "روغن کاری", "روغنکاری"],
        "grease lubrication specified capacity refill G2-LI swing machinery final drive "
    ),
    (
        ["روغن هیدرولیک", "روغن هیدرولیکی", "مخزن روغن"],
        "hydraulic oil tank strainer filter level sight gauge "
    ),
    # ساختار و نگهداری (عمومی)
    (
        ["ساختار و عملکرد", "ساختار و عملکرد و معیار نگهداری", "معیار نگهداری", "نگهداری استاندارد"],
        "structure function and maintenance standard specification reduction ratio "
        "clearance repair limit replace Unit mm "
    ),
    # تست و تنظیم
    (
        ["فشار روغن", "فشار هیدرولیک", "اندازه فشار", "تست فشار"],
        "oil pressure hydraulic pressure measuring testing adjusting relief MPa kg/cm2 "
    ),
    (
        ["عیب یابی", "عیب‌یابی", "خرابی", "مشکل"],
        "troubleshooting failure phenomenon cause standard value in normalcy "
        "Presumed cause references "
    ),
    # دنده و درایو
    (
        ["دنده نهایی", "فاینال درایو", "درایو نهایی", "چرخ زنجیر"],
        "final drive sprocket reduction ratio planetary gear ring gear travel motor "
        "structure function maintenance standard PC800 800LC-8 "
    ),
    # موتور و موتور دیزل
    (
        ["موتور", "موتور دیزل", "انجین", "دیزل"],
        "engine SAA6D140E diesel cooling lubrication specification PC800 800LC-8 "
    ),
    # کد خطا و آلارم
    (
        ["کد خطا", "کد خطا", "فیل کد", "failure code", "آلارم"],
        "failure code CA error troubleshooting diagnostic "
    ),
]
