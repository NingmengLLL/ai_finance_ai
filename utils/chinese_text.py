from __future__ import annotations

from functools import lru_cache


COMMON_T2S_PHRASES = (
    ("騰訊", "腾讯"),
    ("美團", "美团"),
    ("阿里巴巴", "阿里巴巴"),
    ("百度", "百度"),
    ("網易", "网易"),
    ("嗶哩嗶哩", "哔哩哔哩"),
    ("嗶哩", "哔哩"),
    ("拼多多", "拼多多"),
    ("快手", "快手"),
    ("年報", "年报"),
    ("中期報告", "中期报告"),
    ("季度業績", "季度业绩"),
    ("財務報表", "财务报表"),
    ("財務資料", "财务资料"),
    ("綜合收益", "综合收益"),
    ("經營業績", "经营业绩"),
    ("經營活動", "经营活动"),
    ("現金流", "现金流"),
    ("淨利潤", "净利润"),
    ("淨收入", "净收入"),
    ("總收入", "总收入"),
    ("總額", "总额"),
    ("毛利率", "毛利率"),
    ("營業收入", "营业收入"),
    ("營業成本", "营业成本"),
    ("營業利潤", "营业利润"),
    ("銷售", "销售"),
    ("廣告", "广告"),
    ("遊戲", "游戏"),
    ("雲", "云"),
    ("研發", "研发"),
    ("開支", "开支"),
    ("資產", "资产"),
    ("負債", "负债"),
    ("權益", "权益"),
    ("股東", "股东"),
    ("應收", "应收"),
    ("應付", "应付"),
    ("會計", "会计"),
    ("審計", "审计"),
    ("風險", "风险"),
    ("業務", "业务"),
    ("變動", "变动"),
    ("增長", "增长"),
    ("期間", "期间"),
    ("報告", "报告"),
)

COMMON_T2S_CHARS = str.maketrans(
    {
        "與": "与",
        "於": "于",
        "為": "为",
        "業": "业",
        "東": "东",
        "務": "务",
        "報": "报",
        "財": "财",
        "營": "营",
        "銷": "销",
        "淨": "净",
        "現": "现",
        "資": "资",
        "產": "产",
        "負": "负",
        "債": "债",
        "權": "权",
        "益": "益",
        "會": "会",
        "計": "计",
        "審": "审",
        "風": "风",
        "險": "险",
        "變": "变",
        "動": "动",
        "廣": "广",
        "告": "告",
        "遊": "游",
        "戲": "戏",
        "雲": "云",
        "開": "开",
        "發": "发",
        "應": "应",
        "長": "长",
        "總": "总",
        "額": "额",
        "期": "期",
        "間": "间",
    }
)


@lru_cache(maxsize=1)
def _opencc_t2s_converter():
    try:
        from opencc import OpenCC
    except Exception:
        return None

    try:
        return OpenCC("t2s")
    except Exception:
        return None


def normalize_zh_for_retrieval(text: str, mode: str | None = None) -> tuple[str, str]:
    """Return text normalized for Chinese retrieval plus a short status label.

    The project stores source PDFs as-is, but retrieval works better when Hong Kong
    Traditional Chinese filings are embedded in a Simplified Chinese form. OpenCC is
    used when available; the small fallback map only handles common finance terms.
    """

    if mode != "traditional_to_simplified" or not text:
        return text, "not_requested"

    converter = _opencc_t2s_converter()
    if converter is not None:
        return converter.convert(text), "opencc_t2s"

    normalized = text
    for traditional, simplified in COMMON_T2S_PHRASES:
        normalized = normalized.replace(traditional, simplified)
    normalized = normalized.translate(COMMON_T2S_CHARS)

    if normalized == text:
        return text, "opencc_unavailable"
    return normalized, "fallback_common_terms_t2s"
