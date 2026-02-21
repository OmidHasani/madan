# -*- coding: utf-8 -*-
"""
فهرست (TOC) عیب‌یابی: هر کد خطا به محدوده صفحات مربوطش نگاشت می‌شود.
منبع: فهرست مستندات PC800/800LC-8 (Troubleshooting by failure code, E-mode, H-mode, S-mode).

برای هر کد: از صفحه start_page تا end_page (شامل هر دو) = یک بخش کامل عیب‌یابی.
"""

from typing import Dict, List, Optional, Tuple

# لیست مرتب‌شده بر اساس شماره صفحه: (کد, عنوان خلاصه, صفحه شروع)
# end_page برای هر کد = صفحه‌ی شروع کد بعدی منهای ۱
_TOC_ENTRIES: List[Tuple[str, str, int]] = [
    # Troubleshooting by failure code Part 1 (CA, etc.)
    ("CA111", "ECM Critical Internal Failure", 26),
    ("CA115", "Eng Ne and Bkup Speed Sens Error", 28),
    ("CA122", "Chg Air Press Sensor High Error", 30),
    ("CA123", "Chg Air Press Sensor Low Error", 33),
    ("CA131", "Throttle Sensor High Error", 34),
    ("CA132", "Throttle Sensor Low Error", 36),
    ("CA135", "Eng Oil Press Sensor High Error", 38),
    ("CA141", "Eng Oil Press Sensor Low Error", 40),
    ("CA144", "Coolant Temp Sens High Error", 42),
    ("CA145", "Coolant Temp Sens Low Error", 44),
    ("CA153", "Chg Air Temp Sensor High Error", 46),
    ("CA154", "Chg Air Temp Sensor Low Error", 48),
    ("CA187", "Sens Supply 2 Volt Low Error", 49),
    ("CA221", "Ambient Press Sens High Error", 50),
    ("CA222", "Ambient Press Sens Low Error", 52),
    ("CA227", "Sens Supply 2 Volt High Error", 54),
    ("CA234", "Eng Overspeed", 56),
    ("CA238", "Ne Speed Sens Supply Volt Error", 58),
    ("CA263", "Fuel Temp Sensor High Error", 60),
    ("CA265", "Fuel Temp Sensor Low Error", 62),
    ("CA271", "IMV/PCV1 Short Error", 63),
    ("CA272", "IMV/PCV1 Open Error", 64),
    ("CA273", "PCV2 Short Error", 65),
    ("CA274", "PCV2 Open Error", 66),
    ("CA322", "Inj #1 (L#1) Open/Short Error", 67),
    ("CA323", "Inj #5 (L#5) Open/Short Error", 68),
    ("CA324", "Inj #3 (L#3) Open/Short Error", 69),
    ("CA325", "Inj #6 (L#6) Open/Short Error", 70),
    ("CA331", "Inj #2 (L#2) Open/Short Error", 71),
    ("CA332", "Inj #4 (L#4) Open/Short Error", 72),
    ("CA351", "Injectors Drive Circuit Error", 74),
    ("CA352", "Sens Supply 1 Volt Low Error", 76),
    ("CA386", "Sens Supply 1 Volt High Error", 78),
    # Part 2
    ("CA441", "Battery Voltage Low Error", 3),
    ("CA442", "Battery Voltage High Error", 3),
    ("CA449", "Rail Press Very High Error", 4),
    ("CA451", "Rail Press Sensor High Error", 6),
    ("CA452", "Rail Press Sensor Low Error", 8),
    ("CA553", "Rail Press High Error", 8),
    ("CA554", "Rail Press Sensor In Range Error", 9),
    ("CA559", "Rail Press Low Error", 10),
    ("CA689", "Eng Ne Speed Sensor Error", 14),
    ("CA731", "Eng Bkup Speed Sens Phase Error", 16),
    ("CA757", "All Persistent Data Lost Error", 17),
    ("CA778", "Eng Bkup Speed Sensor Error", 18),
    ("CA1228", "EGR Valve Servo Error 1", 20),
    ("CA1625", "EGR Valve Servo Error 2", 21),
    ("CA1626", "BP Valve Sol Current High Error", 22),
    ("CA1627", "BP Valve Sol Current Low Error", 24),
    ("CA1628", "Bypass Valve Servo Error 1", 25),
    ("CA1629", "Bypass Valve Servo Error 2", 26),
    ("CA1631", "BP Valve Pos Sens High Error", 28),
    ("CA1632", "BP Valve Pos Sens Low Error", 30),
    ("CA1633", "KOMNET Datalink Timeout Error", 31),
    ("CA1642", "EGR Inter Press Sens Low Error", 33),
    ("CA2185", "Throt Sens Sup Volt High Error", 36),
    ("CA2186", "Throt Sens Sup Volt Low Error", 38),
    ("CA2249", "Rail Press Very Low Error", 39),
    ("CA2271", "EGR Valve Pos Sens High Error", 40),
    ("CA2272", "EGR Valve Pos Sens Low Error", 42),
    ("CA2351", "EGR Valve Sol Current High Error", 44),
    ("CA2352", "EGR Valve Sol Current Low Error", 46),
    ("CA2555", "Grid Htr Relay Volt Low Error", 47),
    ("CA2556", "Grid Htr Relay Volt High Error", 48),
    ("D110KB", "Battery Relay Drive S/C", 50),
    ("D163KB", "Flash Light Relay S/C", 52),
    ("D195KB", "Step Light Relay S/C", 54),
    ("DA25KP", "Press. Sensor Power Abnormality", 56),
    ("DA2SKQ", "Model Selection Abnormality", 58),
    ("DA80MA", "Auto. Lub. Abnormal", 60),
    ("DA2RMC", "Pump Comm. Abnormality", 62),
    ("DAFRMC", "Monitor Comm. Abnormality", 64),
    ("DGE5KY", "Ambi. Temp. Sensor S/C", 66),
    ("DGH2KB", "Hydr. Oil Temp. Sensor S/C", 68),
    # Part 3
    ("DH25KA", "L Jet Sensor Disc", 4),
    ("DH25KB", "L Jet Sensor S/C", 6),
    ("DH26KA", "R Jet Sensor Disc", 8),
    ("DH26KB", "R Jet Sensor S/C", 10),
    ("DHPEKA", "F Pump P. Sensor Disc", 12),
    ("DHPEKB", "F Pump P. Sensor S/C", 14),
    ("DHPFKA", "R Pump P. Sensor Disc", 16),
    ("DHPFKB", "R Pump P. Sensor S/C", 18),
    ("DV20KB", "Travel Alarm S/C", 20),
    ("DW41KA", "Swing Priority Sol. Disc", 22),
    ("DW41KB", "Swing Priority Sol. S/C", 24),
    ("DW43KA", "Travel Speed Sol. Disc", 26),
    ("DW43KB", "Travel Speed Sol. S/C", 28),
    ("DW45KA", "Swing Brake Sol. Disc", 30),
    ("DW45KB", "Swing Brake Sol. S/C", 32),
    ("DW7BKA", "Fan Reverse Sol. Disc", 34),
    ("DW78KB", "Fan Reverse Sol. S/C", 36),
    ("DWK0KA", "2-stage Relief Sol. Disc", 38),
    ("DWK0KB", "2-stage Relief Sol. S/C", 40),
    ("DX16KA", "Fan Pump EPC Sol. Disc", 42),
    ("DX16KB", "Fan Pump EPC Sol. S/C", 44),
    ("DXAAKA", "F Pump EPC Sol. Disc", 46),
    ("DXAAKB", "F Pump EPC Sol. S/C", 48),
    ("DXABKA", "R Pump EPC Sol. Disc", 50),
    ("DXABKB", "R Pump EPC Sol. S/C", 52),
    ("DY20KA", "Wiper Working Abnormality", 54),
    ("DY20MA", "Wiper Parking Abnormality", 56),
    ("DY2CKB", "Washer Drive S/C", 60),
    ("DY2DKB", "Wiper Drive (For) S/C", 62),
    ("DY2EKB", "Wiper Drive (Rev) S/C", 66),
    # E-mode (electrical)
    ("E-1", "Engine does not start (Engine does not rotate)", 6),
    ("E-2", "Preheater does not operate", 9),
    ("E-3", "Auto engine warm-up device does not work", 14),
    ("E-4", "Auto-decelerator does not operate", 15),
    ("E-5", "All work equipment, swing and travel do not move", 16),
    ("E-6", "Machine push-up function does not operate normally", 18),
    ("E-7", "Boom shockless function does not operate normally", 20),
    ("E-8", "Any item is not displayed on machine monitor", 22),
    ("E-9", "Part of display on machine monitor is missing", 23),
    ("E-10", "Machine monitor displays contents irrelevant to the model", 23),
    ("E-11", "Fuel level monitor red lamp lights up while engine is running", 24),
    ("E-12", "Engine coolant thermometer does not display normally", 26),
    ("E-13", "Hydraulic oil temperature gauge does not display correctly", 28),
    ("E-14", "Fuel gauge does not display correctly", 29),
    ("E-15", "Swing lock monitor does not display correctly", 30),
    ("E-16", "When monitor switch is operated, nothing is displayed", 32),
    ("E-17", "Windshield wiper and window washer do not work", 34),
    ("E-18", "Boom RAISE not correctly displayed in monitor function", 42),
    ("E-19", "Boom LOWER not correctly displayed in monitor function", 43),
    ("E-20", "Arm IN not correctly displayed in monitor function", 44),
    ("E-21", "Arm OUT not correctly displayed in monitor function", 45),
    ("E-22", "Bucket CURL not correctly displayed in monitor function", 46),
    ("E-23", "Bucket DUMP not correctly displayed in monitor function", 47),
    ("E-24", "SWING not correctly displayed in monitor function", 48),
    ("E-25", "Left travel not displayed normally in monitoring function", 50),
    ("E-26", "Right travel not displayed normally in monitoring function", 52),
    ("E-27", "Service not correctly displayed in monitor function", 54),
    ("E-28", "KOMTRAX system does not operate normally", 56),
    ("E-29", "Air conditioner does not work", 58),
    ("E-30", "Step light does not light up or go off", 59),
    ("E-31", "Electric grease gun does not operate", 62),
    ("E-32", "Travel alarm does not sound or does not stop sounding", 64),
    # H-mode (hydraulic/mechanical)
    ("H-1", "Speed or power of all work equipment, travel, and swing is low", 4),
    ("H-2", "Engine speed lowers remarkably or engine stalls", 6),
    ("H-3", "All work equipment, travel, and swing systems do not work", 7),
    ("H-4", "Abnormal sound is heard from around pump", 8),
    ("H-5", "Boom speed or power is low", 9),
    ("H-6", "Speed or power of arm is low", 11),
    ("H-7", "Speed or power of bucket is low", 12),
    ("H-8", "Boom does not move", 13),
    ("H-9", "Arm does not move", 13),
    ("H-10", "Bucket does not move", 13),
    ("H-11", "Hydraulic drift of work equipment is large", 14),
    ("H-12", "Time lag of work equipment is large", 16),
    ("H-13", "Heavy lift function does not operate or stop", 17),
    ("H-14", "Machine push-up function does not operate or stop", 17),
    ("H-15", "Boom shockless function cannot be turned ON or OFF", 17),
    ("H-16", "Machine deviates in one direction", 18),
    ("H-17", "Machine deviates largely at start", 20),
    ("H-18", "Machine deviates largely during compound operation", 21),
    ("H-19", "Travel speed or power is low", 21),
    ("H-20", "Machine does not travel (only one track)", 22),
    ("H-21", "Travel speed does not change", 23),
    ("H-22", "Upper structure does not swing", 24),
    ("H-23", "Swing speed or acceleration is low", 26),
    ("H-24", "Swing speed or acceleration is low during compound operation", 27),
    ("H-25", "Upper structure overruns excessively when it stops swinging", 29),
    ("H-26", "Large shock when upper structure stops swinging", 30),
    ("H-27", "Large abnormal sound when upper structure stops swinging", 31),
    ("H-28", "Hydraulic drift of swing is large", 32),
    # S-mode (engine)
    ("S-1", "Starting performance is poor", 6),
    ("S-2", "Engine does not start", 8),
    ("S-3", "Engine does not pick up smoothly", 12),
    ("S-4", "Engine stops during operations", 13),
    ("S-6", "Engine lacks output (or lacks power)", 15),
    ("S-7", "Exhaust gas color is black (incomplete combustion)", 16),
    ("S-8", "Oil consumption is excessive (or exhaust smoke is blue)", 18),
    ("S-9", "Oil becomes contaminated quickly", 19),
    ("S-10", "Fuel consumption is excessive", 20),
    ("S-11", "Oil is in coolant (or coolant spurts back or coolant level goes down)", 21),
    ("S-12", "Oil pressure drops", 22),
    ("S-13", "Oil level rises (Entry of coolant/fuel)", 23),
    ("S-14", "Coolant temperature becomes too high (overheating)", 25),
    ("S-15", "Abnormal noise is made", 26),
    ("S-16", "Vibration is excessive", 26),
]

# نرمال‌سازی کد برای تطابق: H-22 و H22 هر دو به یک کلید بروند
def _norm_code(raw: str) -> str:
    s = (raw or "").strip().upper()
    # E-31, H-22, CA1626
    if "-" not in s and len(s) >= 2 and s[0].isalpha() and s[1:].replace(" ", "").isdigit():
        # H22 -> H-22, E31 -> E-31
        for i, c in enumerate(s):
            if c.isdigit():
                return s[:i] + "-" + s[i:] if i > 0 else s
    return s.replace(" ", "")


def _build_toc_maps() -> Tuple[Dict[str, Tuple[int, int, str]], Dict[int, str]]:
    """ساخت نگاشت کد -> (start_page, end_page, title) و صفحه -> کد (برای استنتاج از جستجو)."""
    entries = sorted(_TOC_ENTRIES, key=lambda x: (x[2], x[0]))
    code_to_range: Dict[str, Tuple[int, int, str]] = {}
    page_to_code: Dict[int, str] = {}

    for i, (code, title, start_page) in enumerate(entries):
        norm = _norm_code(code)
        end_page = start_page
        if i + 1 < len(entries):
            next_start = entries[i + 1][2]
            if next_start > start_page:
                end_page = next_start - 1
            # اگر کد بعدی همان صفحه است، end_page = start_page (فقط همان یک صفحه)
        code_to_range[norm] = (start_page, end_page, title)
        for p in range(start_page, end_page + 1):
            page_to_code[p] = norm
        alt = code.upper().replace("-", "")
        if alt != norm:
            code_to_range.setdefault(alt, (start_page, end_page, title))
    return code_to_range, page_to_code


_CODE_TO_RANGE: Dict[str, Tuple[int, int, str]] = {}
_PAGE_TO_CODE: Dict[int, str] = {}


def _init_toc() -> None:
    global _CODE_TO_RANGE, _PAGE_TO_CODE
    if _CODE_TO_RANGE:
        return
    _CODE_TO_RANGE, _PAGE_TO_CODE = _build_toc_maps()
    # نرمال‌سازی کلیدهای code_to_range برای تطابق با فرمت‌های مختلف
    add = {}
    for k, v in list(_CODE_TO_RANGE.items()):
        n = _norm_code(k)
        if n != k:
            add[n] = v
    for k, v in add.items():
        _CODE_TO_RANGE.setdefault(k, v)


def get_section_page_range(code: str) -> Optional[Tuple[int, int, str]]:
    """
    برای کد خطا (مثل E-31، H-22، CA1626) محدوده صفحات و عنوان را برمی‌گرداند.
    Returns:
        (start_page, end_page, title) یا None اگر کد در TOC نبود.
    """
    _init_toc()
    norm = _norm_code(code)
    return _CODE_TO_RANGE.get(norm) or _CODE_TO_RANGE.get(code.upper()) or _CODE_TO_RANGE.get(code.upper().replace("-", ""))


def get_code_for_page(page: int) -> Optional[str]:
    """برای یک شماره صفحه، کد خطای مربوط به آن بخش را برمی‌گرداند (برای استنتاج از نتایج جستجو)."""
    _init_toc()
    return _PAGE_TO_CODE.get(page)


def get_all_codes() -> List[str]:
    """لیست تمام کدهای شناخته‌شده (نرمال‌شده)."""
    _init_toc()
    seen = set()
    out = []
    for code, _, _ in _TOC_ENTRIES:
        n = _norm_code(code)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return sorted(out)
