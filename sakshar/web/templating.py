from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from ..config import LANGUAGES, get_settings
from ..pipeline.run import STAGES

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")
templates.env.globals.update(LANGUAGES=LANGUAGES, STAGES=STAGES, settings=get_settings)

UI = {
    "en": {"explain": "Explanation", "fields": "Key details", "ask": "Ask", "review": "Please check",
           "confirm": "Looks right", "fix": "Correct it", "unverified": "Not verified against the document",
           "next": "What to do next", "placeholder": "Ask a question about this document...", "send": "Send",
           "hold": "Tap to speak", "stop": "Tap to stop", "listen": "Listen"},
    "hi": {"explain": "व्याख्या", "fields": "मुख्य जानकारी", "ask": "पूछें", "review": "कृपया जाँचें",
           "confirm": "सही है", "fix": "सुधारें", "unverified": "दस्तावेज़ से पुष्टि नहीं हुई",
           "next": "आगे क्या करें", "placeholder": "इस दस्तावेज़ के बारे में प्रश्न पूछें...", "send": "भेजें",
           "hold": "बोलने के लिए दबाएँ", "stop": "रोकने के लिए दबाएँ", "listen": "सुनें"},
    "mr": {"explain": "स्पष्टीकरण", "fields": "मुख्य माहिती", "ask": "विचारा", "review": "कृपया तपासा",
           "confirm": "बरोबर आहे", "fix": "दुरुस्त करा", "unverified": "कागदपत्राशी पडताळणी झाली नाही",
           "next": "पुढे काय करावे", "placeholder": "या कागदपत्राबद्दल प्रश्न विचारा...", "send": "पाठवा",
           "hold": "बोलण्यासाठी दाबा", "stop": "थांबवण्यासाठी दाबा", "listen": "ऐका"},
    "pa": {"explain": "ਵਿਆਖਿਆ", "fields": "ਮੁੱਖ ਜਾਣਕਾਰੀ", "ask": "ਪੁੱਛੋ", "review": "ਕਿਰਪਾ ਕਰਕੇ ਜਾਂਚੋ",
           "confirm": "ਸਹੀ ਹੈ", "fix": "ਠੀਕ ਕਰੋ", "unverified": "ਦਸਤਾਵੇਜ਼ ਨਾਲ ਪੁਸ਼ਟੀ ਨਹੀਂ ਹੋਈ",
           "next": "ਅੱਗੇ ਕੀ ਕਰਨਾ ਹੈ", "placeholder": "ਇਸ ਦਸਤਾਵੇਜ਼ ਬਾਰੇ ਸਵਾਲ ਪੁੱਛੋ...", "send": "ਭੇਜੋ",
           "hold": "ਬੋਲਣ ਲਈ ਦਬਾਓ", "stop": "ਰੋਕਣ ਲਈ ਦਬਾਓ", "listen": "ਸੁਣੋ"},
}
templates.env.globals["UI"] = UI
