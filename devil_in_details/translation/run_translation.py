import argparse
from typing import List
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# sys.path.append("/home/bee82nf/devil-in-details")
from devil_in_details.utils import load_jsonl, save_jsonl
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ISO2NLLB = {
    "ace": {"name": "Acehnese (Latin script)", "code": "ace_Latn"},
    "acm": {"name": "Mesopotamian Arabic", "code": "acm_Arab"},
    "ar": {"name": "Arabic", "code": "arb_Arab"},
    "acq": {"name": "Ta’izzi-Adeni Arabic", "code": "acq_Arab"},
    "aeb": {"name": "Tunisian Arabic", "code": "aeb_Arab"},
    "afr": {"name": "Afrikaans", "code": "afr_Latn"},
    "af": {"name": "Afrikaans", "code": "afr_Latn"},
    "ajp": {"name": "South Levantine Arabic", "code": "ajp_Arab"},
    "aka": {"name": "Akan", "code": "aka_Latn"},
    "ak": {"name": "Akan", "code": "aka_Latn"},
    "amh": {"name": "Amharic", "code": "amh_Ethi"},
    "am": {"name": "Amharic", "code": "amh_Ethi"},
    "apc": {"name": "North Levantine Arabic", "code": "apc_Arab"},
    "arb": {"name": "Modern Standard Arabic (Romanized)", "code": "arb_Latn"},
    "ars": {"name": "Najdi Arabic", "code": "ars_Arab"},
    "ary": {"name": "Moroccan Arabic", "code": "ary_Arab"},
    "arz": {"name": "Egyptian Arabic", "code": "arz_Arab"},
    "asm": {"name": "Assamese", "code": "asm_Beng"},
    "as": {"name": "Assamese", "code": "asm_Beng"},
    "ast": {"name": "Asturian", "code": "ast_Latn"},
    "awa": {"name": "Awadhi", "code": "awa_Deva"},
    "ayr": {"name": "Central Aymara", "code": "ayr_Latn"},
    "ay": {"name": "Aymara", "code": "ayr_Latn"},
    "azb": {"name": "South Azerbaijani", "code": "azb_Arab"},
    "az": {"name": "Azerbaijani", "code": "azj_Latn"},
    "azj": {"name": "North Azerbaijani", "code": "azj_Latn"},
    "bak": {"name": "Bashkir", "code": "bak_Cyrl"},
    "ba": {"name": "Bashkir", "code": "bak_Cyrl"},
    "bam": {"name": "Bambara", "code": "bam_Latn"},
    "bm": {"name": "Bambara", "code": "bam_Latn"},
    "ban": {"name": "Balinese", "code": "ban_Latn"},
    "bel": {"name": "Belarusian", "code": "bel_Cyrl"},
    "be": {"name": "Belarusian", "code": "bel_Cyrl"},
    "bem": {"name": "Bemba", "code": "bem_Latn"},
    "ben": {"name": "Bengali", "code": "ben_Beng"},
    "bn": {"name": "Bengali", "code": "ben_Beng"},
    "bho": {"name": "Bhojpuri", "code": "bho_Deva"},
    "bjn": {"name": "Banjar (Latin script)", "code": "bjn_Latn"},
    "ms": {"name": "Malay (macrolanguage)", "code": "zsm_Latn"},
    "bod": {"name": "Standard Tibetan", "code": "bod_Tibt"},
    "bo": {"name": "Tibetan", "code": "bod_Tibt"},
    "bos": {"name": "Bosnian", "code": "bos_Latn"},
    "bs": {"name": "Bosnian", "code": "bos_Latn"},
    "bug": {"name": "Buginese", "code": "bug_Latn"},
    "bul": {"name": "Bulgarian", "code": "bul_Cyrl"},
    "bg": {"name": "Bulgarian", "code": "bul_Cyrl"},
    "cat": {"name": "Catalan", "code": "cat_Latn"},
    "ca": {"name": "Catalan", "code": "cat_Latn"},
    "ceb": {"name": "Cebuano", "code": "ceb_Latn"},
    "ces": {"name": "Czech", "code": "ces_Latn"},
    "cs": {"name": "Czech", "code": "ces_Latn"},
    "cjk": {"name": "Chokwe", "code": "cjk_Latn"},
    "ckb": {"name": "Central Kurdish", "code": "ckb_Arab"},
    "ku": {"name": "Kurdish", "code": "ckb_Arab"},
    "crh": {"name": "Crimean Tatar", "code": "crh_Latn"},
    "cym": {"name": "Welsh", "code": "cym_Latn"},
    "cy": {"name": "Welsh", "code": "cym_Latn"},
    "dan": {"name": "Danish", "code": "dan_Latn"},
    "da": {"name": "Danish", "code": "dan_Latn"},
    "deu": {"name": "German", "code": "deu_Latn"},
    "de": {"name": "German", "code": "deu_Latn"},
    "de-st": {"name": "Southern Tyrolean", "code": "deu_Latn"},
    "dik": {"name": "Southwestern Dinka", "code": "dik_Latn"},
    "dyu": {"name": "Dyula", "code": "dyu_Latn"},
    "dzo": {"name": "Dzongkha", "code": "dzo_Tibt"},
    "dz": {"name": "Dzongkha", "code": "dzo_Tibt"},
    "ell": {"name": "Greek", "code": "ell_Grek"},
    "el": {"name": "Modern Greek (1453-)", "code": "ell_Grek"},
    "eng": {"name": "English", "code": "eng_Latn"},
    "en": {"name": "English", "code": "eng_Latn"},
    "epo": {"name": "Esperanto", "code": "epo_Latn"},
    "eo": {"name": "Esperanto", "code": "epo_Latn"},
    "est": {"name": "Estonian", "code": "est_Latn"},
    "et": {"name": "Estonian", "code": "est_Latn"},
    "eus": {"name": "Basque", "code": "eus_Latn"},
    "eu": {"name": "Basque", "code": "eus_Latn"},
    "ewe": {"name": "Ewe", "code": "ewe_Latn"},
    "ee": {"name": "Ewe", "code": "ewe_Latn"},
    "fao": {"name": "Faroese", "code": "fao_Latn"},
    "fo": {"name": "Faroese", "code": "fao_Latn"},
    "fij": {"name": "Fijian", "code": "fij_Latn"},
    "fj": {"name": "Fijian", "code": "fij_Latn"},
    "fin": {"name": "Finnish", "code": "fin_Latn"},
    "fi": {"name": "Finnish", "code": "fin_Latn"},
    "fon": {"name": "Fon", "code": "fon_Latn"},
    "fra": {"name": "French", "code": "fra_Latn"},
    "fr": {"name": "French", "code": "fra_Latn"},
    "fur": {"name": "Friulian", "code": "fur_Latn"},
    "fuv": {"name": "Nigerian Fulfulde", "code": "fuv_Latn"},
    "ff": {"name": "Fulah", "code": "fuv_Latn"},
    "gla": {"name": "Scottish Gaelic", "code": "gla_Latn"},
    "gd": {"name": "Scottish Gaelic", "code": "gla_Latn"},
    "gle": {"name": "Irish", "code": "gle_Latn"},
    "ga": {"name": "Irish", "code": "gle_Latn"},
    "glg": {"name": "Galician", "code": "glg_Latn"},
    "gl": {"name": "Galician", "code": "glg_Latn"},
    "grn": {"name": "Guarani", "code": "grn_Latn"},
    "gn": {"name": "Guarani", "code": "grn_Latn"},
    "guj": {"name": "Gujarati", "code": "guj_Gujr"},
    "gu": {"name": "Gujarati", "code": "guj_Gujr"},
    "hat": {"name": "Haitian Creole", "code": "hat_Latn"},
    "ht": {"name": "Haitian", "code": "hat_Latn"},
    "hau": {"name": "Hausa", "code": "hau_Latn"},
    "ha": {"name": "Hausa", "code": "hau_Latn"},
    "heb": {"name": "Hebrew", "code": "heb_Hebr"},
    "he": {"name": "Hebrew", "code": "heb_Hebr"},
    "hin": {"name": "Hindi", "code": "hin_Deva"},
    "hi": {"name": "Hindi", "code": "hin_Deva"},
    "hne": {"name": "Chhattisgarhi", "code": "hne_Deva"},
    "hrv": {"name": "Croatian", "code": "hrv_Latn"},
    "hr": {"name": "Croatian", "code": "hrv_Latn"},
    "hun": {"name": "Hungarian", "code": "hun_Latn"},
    "hu": {"name": "Hungarian", "code": "hun_Latn"},
    "hye": {"name": "Armenian", "code": "hye_Armn"},
    "hy": {"name": "Armenian", "code": "hye_Armn"},
    "ibo": {"name": "Igbo", "code": "ibo_Latn"},
    "ig": {"name": "Igbo", "code": "ibo_Latn"},
    "ilo": {"name": "Ilocano", "code": "ilo_Latn"},
    "ind": {"name": "Indonesian", "code": "ind_Latn"},
    "id": {"name": "Indonesian", "code": "ind_Latn"},
    "isl": {"name": "Icelandic", "code": "isl_Latn"},
    "is": {"name": "Icelandic", "code": "isl_Latn"},
    "ita": {"name": "Italian", "code": "ita_Latn"},
    "it": {"name": "Italian", "code": "ita_Latn"},
    "jav": {"name": "Javanese", "code": "jav_Latn"},
    "jv": {"name": "Javanese", "code": "jav_Latn"},
    "jpn": {"name": "Japanese", "code": "jpn_Jpan"},
    "ja": {"name": "Japanese", "code": "jpn_Jpan"},
    "kab": {"name": "Kabyle", "code": "kab_Latn"},
    "kac": {"name": "Jingpho", "code": "kac_Latn"},
    "kam": {"name": "Kamba", "code": "kam_Latn"},
    "kan": {"name": "Kannada", "code": "kan_Knda"},
    "kn": {"name": "Kannada", "code": "kan_Knda"},
    "kas": {"name": "Kashmiri (Devanagari script)", "code": "kas_Deva"},
    "ks": {"name": "Kashmiri", "code": "kas_Arab"},
    "kat": {"name": "Georgian", "code": "kat_Geor"},
    "ka": {"name": "Georgian", "code": "kat_Geor"},
    "knc": {"name": "Central Kanuri (Latin script)", "code": "knc_Latn"},
    "kr": {"name": "Kanuri", "code": "knc_Arab"},
    "kaz": {"name": "Kazakh", "code": "kaz_Cyrl"},
    "kk": {"name": "Kazakh", "code": "kaz_Cyrl"},
    "kbp": {"name": "Kabiyè", "code": "kbp_Latn"},
    "kea": {"name": "Kabuverdianu", "code": "kea_Latn"},
    "khm": {"name": "Khmer", "code": "khm_Khmr"},
    "km": {"name": "Khmer", "code": "khm_Khmr"},
    "kik": {"name": "Kikuyu", "code": "kik_Latn"},
    "ki": {"name": "Kikuyu", "code": "kik_Latn"},
    "kin": {"name": "Kinyarwanda", "code": "kin_Latn"},
    "rw": {"name": "Kinyarwanda", "code": "kin_Latn"},
    "kir": {"name": "Kyrgyz", "code": "kir_Cyrl"},
    "ky": {"name": "Kirghiz", "code": "kir_Cyrl"},
    "kmb": {"name": "Kimbundu", "code": "kmb_Latn"},
    "kmr": {"name": "Northern Kurdish", "code": "kmr_Latn"},
    "kon": {"name": "Kikongo", "code": "kon_Latn"},
    "kg": {"name": "Kongo", "code": "kon_Latn"},
    "kor": {"name": "Korean", "code": "kor_Hang"},
    "ko": {"name": "Korean", "code": "kor_Hang"},
    "lao": {"name": "Lao", "code": "lao_Laoo"},
    "lo": {"name": "Lao", "code": "lao_Laoo"},
    "lij": {"name": "Ligurian", "code": "lij_Latn"},
    "lim": {"name": "Limburgish", "code": "lim_Latn"},
    "li": {"name": "Limburgan", "code": "lim_Latn"},
    "lin": {"name": "Lingala", "code": "lin_Latn"},
    "ln": {"name": "Lingala", "code": "lin_Latn"},
    "lit": {"name": "Lithuanian", "code": "lit_Latn"},
    "lt": {"name": "Lithuanian", "code": "lit_Latn"},
    "lmo": {"name": "Lombard", "code": "lmo_Latn"},
    "ltg": {"name": "Latgalian", "code": "ltg_Latn"},
    "lv": {"name": "Latvian", "code": "lvs_Latn"},
    "ltz": {"name": "Luxembourgish", "code": "ltz_Latn"},
    "lb": {"name": "Luxembourgish", "code": "ltz_Latn"},
    "lua": {"name": "Luba-Kasai", "code": "lua_Latn"},
    "lug": {"name": "Ganda", "code": "lug_Latn"},
    "lg": {"name": "Ganda", "code": "lug_Latn"},
    "luo": {"name": "Luo", "code": "luo_Latn"},
    "lus": {"name": "Mizo", "code": "lus_Latn"},
    "lvs": {"name": "Standard Latvian", "code": "lvs_Latn"},
    "mag": {"name": "Magahi", "code": "mag_Deva"},
    "mai": {"name": "Maithili", "code": "mai_Deva"},
    "mal": {"name": "Malayalam", "code": "mal_Mlym"},
    "ml": {"name": "Malayalam", "code": "mal_Mlym"},
    "mar": {"name": "Marathi", "code": "mar_Deva"},
    "mr": {"name": "Marathi", "code": "mar_Deva"},
    "min": {"name": "Minangkabau (Latin script)", "code": "min_Latn"},
    "mkd": {"name": "Macedonian", "code": "mkd_Cyrl"},
    "mk": {"name": "Macedonian", "code": "mkd_Cyrl"},
    "plt": {"name": "Plateau Malagasy", "code": "plt_Latn"},
    "mg": {"name": "Malagasy", "code": "plt_Latn"},
    "mlt": {"name": "Maltese", "code": "mlt_Latn"},
    "mt": {"name": "Maltese", "code": "mlt_Latn"},
    "mni": {"name": "Meitei (Bengali script)", "code": "mni_Beng"},
    "khk": {"name": "Halh Mongolian", "code": "khk_Cyrl"},
    "mn": {"name": "Mongolian", "code": "khk_Cyrl"},
    "mos": {"name": "Mossi", "code": "mos_Latn"},
    "mri": {"name": "Maori", "code": "mri_Latn"},
    "mi": {"name": "Maori", "code": "mri_Latn"},
    "mya": {"name": "Burmese", "code": "mya_Mymr"},
    "my": {"name": "Burmese", "code": "mya_Mymr"},
    "nld": {"name": "Dutch", "code": "nld_Latn"},
    "nl": {"name": "Dutch", "code": "nld_Latn"},
    "nno": {"name": "Norwegian Nynorsk", "code": "nno_Latn"},
    "nn": {"name": "Norwegian Nynorsk", "code": "nno_Latn"},
    "nob": {"name": "Norwegian Bokmål", "code": "nob_Latn"},
    "nb": {"name": "Norwegian BokmÃ¥l", "code": "nob_Latn"},
    "npi": {"name": "Nepali", "code": "npi_Deva"},
    "ne": {"name": "Nepali (macrolanguage)", "code": "npi_Deva"},
    "nso": {"name": "Northern Sotho", "code": "nso_Latn"},
    "nus": {"name": "Nuer", "code": "nus_Latn"},
    "nya": {"name": "Nyanja", "code": "nya_Latn"},
    "ny": {"name": "Nyanja", "code": "nya_Latn"},
    "oci": {"name": "Occitan", "code": "oci_Latn"},
    "oc": {"name": "Occitan (post 1500)", "code": "oci_Latn"},
    "gaz": {"name": "West Central Oromo", "code": "gaz_Latn"},
    "om": {"name": "Oromo", "code": "gaz_Latn"},
    "ory": {"name": "Odia", "code": "ory_Orya"},
    "or": {"name": "Oriya (macrolanguage)", "code": "ory_Orya"},
    "pag": {"name": "Pangasinan", "code": "pag_Latn"},
    "pan": {"name": "Eastern Panjabi", "code": "pan_Guru"},
    "pa": {"name": "Panjabi", "code": "pan_Guru"},
    "pap": {"name": "Papiamento", "code": "pap_Latn"},
    "pes": {"name": "Western Persian", "code": "pes_Arab"},
    "fa": {"name": "Persian", "code": "pes_Arab"},
    "pol": {"name": "Polish", "code": "pol_Latn"},
    "pl": {"name": "Polish", "code": "pol_Latn"},
    "por": {"name": "Portuguese", "code": "por_Latn"},
    "pt": {"name": "Portuguese", "code": "por_Latn"},
    "prs": {"name": "Dari", "code": "prs_Arab"},
    "pbt": {"name": "Southern Pashto", "code": "pbt_Arab"},
    "ps": {"name": "Pushto", "code": "pbt_Arab"},
    "quy": {"name": "Ayacucho Quechua", "code": "quy_Latn"},
    "qu": {"name": "Quechua", "code": "quy_Latn"},
    "ron": {"name": "Romanian", "code": "ron_Latn"},
    "ro": {"name": "Romanian", "code": "ron_Latn"},
    "run": {"name": "Rundi", "code": "run_Latn"},
    "rn": {"name": "Rundi", "code": "run_Latn"},
    "rus": {"name": "Russian", "code": "rus_Cyrl"},
    "ru": {"name": "Russian", "code": "rus_Cyrl"},
    "sag": {"name": "Sango", "code": "sag_Latn"},
    "sg": {"name": "Sango", "code": "sag_Latn"},
    "san": {"name": "Sanskrit", "code": "san_Deva"},
    "sa": {"name": "Sanskrit", "code": "san_Deva"},
    "sat": {"name": "Santali", "code": "sat_Olck"},
    "scn": {"name": "Sicilian", "code": "scn_Latn"},
    "shn": {"name": "Shan", "code": "shn_Mymr"},
    "sin": {"name": "Sinhala", "code": "sin_Sinh"},
    "si": {"name": "Sinhala", "code": "sin_Sinh"},
    "slk": {"name": "Slovak", "code": "slk_Latn"},
    "sk": {"name": "Slovak", "code": "slk_Latn"},
    "slv": {"name": "Slovenian", "code": "slv_Latn"},
    "sl": {"name": "Slovenian", "code": "slv_Latn"},
    "smo": {"name": "Samoan", "code": "smo_Latn"},
    "sm": {"name": "Samoan", "code": "smo_Latn"},
    "sna": {"name": "Shona", "code": "sna_Latn"},
    "sn": {"name": "Shona", "code": "sna_Latn"},
    "snd": {"name": "Sindhi", "code": "snd_Arab"},
    "sd": {"name": "Sindhi", "code": "snd_Arab"},
    "som": {"name": "Somali", "code": "som_Latn"},
    "so": {"name": "Somali", "code": "som_Latn"},
    "sot": {"name": "Southern Sotho", "code": "sot_Latn"},
    "st": {"name": "Southern Sotho", "code": "sot_Latn"},
    "spa": {"name": "Spanish", "code": "spa_Latn"},
    "es": {"name": "Spanish", "code": "spa_Latn"},
    "als": {"name": "Tosk Albanian", "code": "als_Latn"},
    "sq": {"name": "Albanian", "code": "als_Latn"},
    "srd": {"name": "Sardinian", "code": "srd_Latn"},
    "sc": {"name": "Sardinian", "code": "srd_Latn"},
    "srp": {"name": "Serbian", "code": "srp_Cyrl"},
    "sr": {"name": "Serbian", "code": "srp_Cyrl"},
    "ssw": {"name": "Swati", "code": "ssw_Latn"},
    "ss": {"name": "Swati", "code": "ssw_Latn"},
    "sun": {"name": "Sundanese", "code": "sun_Latn"},
    "su": {"name": "Sundanese", "code": "sun_Latn"},
    "swe": {"name": "Swedish", "code": "swe_Latn"},
    "sv": {"name": "Swedish", "code": "swe_Latn"},
    "swh": {"name": "Swahili", "code": "swh_Latn"},
    "sw": {"name": "Swahili (macrolanguage)", "code": "swh_Latn"},
    "szl": {"name": "Silesian", "code": "szl_Latn"},
    "tam": {"name": "Tamil", "code": "tam_Taml"},
    "ta": {"name": "Tamil", "code": "tam_Taml"},
    "tat": {"name": "Tatar", "code": "tat_Cyrl"},
    "tt": {"name": "Tatar", "code": "tat_Cyrl"},
    "tel": {"name": "Telugu", "code": "tel_Telu"},
    "te": {"name": "Telugu", "code": "tel_Telu"},
    "tgk": {"name": "Tajik", "code": "tgk_Cyrl"},
    "tg": {"name": "Tajik", "code": "tgk_Cyrl"},
    "tgl": {"name": "Tagalog", "code": "tgl_Latn"},
    "tl": {"name": "Tagalog", "code": "tgl_Latn"},
    "tha": {"name": "Thai", "code": "tha_Thai"},
    "th": {"name": "Thai", "code": "tha_Thai"},
    "tir": {"name": "Tigrinya", "code": "tir_Ethi"},
    "ti": {"name": "Tigrinya", "code": "tir_Ethi"},
    "taq": {"name": "Tamasheq (Tifinagh script)", "code": "taq_Tfng"},
    "tpi": {"name": "Tok Pisin", "code": "tpi_Latn"},
    "tsn": {"name": "Tswana", "code": "tsn_Latn"},
    "tn": {"name": "Tswana", "code": "tsn_Latn"},
    "tso": {"name": "Tsonga", "code": "tso_Latn"},
    "ts": {"name": "Tsonga", "code": "tso_Latn"},
    "tuk": {"name": "Turkmen", "code": "tuk_Latn"},
    "tk": {"name": "Turkmen", "code": "tuk_Latn"},
    "tum": {"name": "Tumbuka", "code": "tum_Latn"},
    "tur": {"name": "Turkish", "code": "tur_Latn"},
    "tr": {"name": "Turkish", "code": "tur_Latn"},
    "twi": {"name": "Twi", "code": "twi_Latn"},
    "tw": {"name": "Twi", "code": "twi_Latn"},
    "tzm": {"name": "Central Atlas Tamazight", "code": "tzm_Tfng"},
    "uig": {"name": "Uyghur", "code": "uig_Arab"},
    "ug": {"name": "Uighur", "code": "uig_Arab"},
    "ukr": {"name": "Ukrainian", "code": "ukr_Cyrl"},
    "uk": {"name": "Ukrainian", "code": "ukr_Cyrl"},
    "umb": {"name": "Umbundu", "code": "umb_Latn"},
    "urd": {"name": "Urdu", "code": "urd_Arab"},
    "ur": {"name": "Urdu", "code": "urd_Arab"},
    "uzn": {"name": "Northern Uzbek", "code": "uzn_Latn"},
    "uz": {"name": "Uzbek", "code": "uzn_Latn"},
    "vec": {"name": "Venetian", "code": "vec_Latn"},
    "vie": {"name": "Vietnamese", "code": "vie_Latn"},
    "vi": {"name": "Vietnamese", "code": "vie_Latn"},
    "war": {"name": "Waray", "code": "war_Latn"},
    "wol": {"name": "Wolof", "code": "wol_Latn"},
    "wo": {"name": "Wolof", "code": "wol_Latn"},
    "xho": {"name": "Xhosa", "code": "xho_Latn"},
    "xh": {"name": "Xhosa", "code": "xho_Latn"},
    "ydd": {"name": "Eastern Yiddish", "code": "ydd_Hebr"},
    "yi": {"name": "Yiddish", "code": "ydd_Hebr"},
    "yor": {"name": "Yoruba", "code": "yor_Latn"},
    "yo": {"name": "Yoruba", "code": "yor_Latn"},
    "yue": {"name": "Yue Chinese", "code": "yue_Hant"},
    "zh": {"name": "Chinese", "code": "zho_Hant"},
    "zho": {"name": "Chinese (Traditional)", "code": "zho_Hant"},
    "zsm": {"name": "Standard Malay", "code": "zsm_Latn"},
    "zul": {"name": "Zulu", "code": "zul_Latn"},
    "zu": {"name": "Zulu", "code": "zul_Latn"},
    "aym": {"name": "Aymara", "code": "ayr_Latn"},
    "swa": {"name": "Swahili", "code": "swh_Latn"},
}


class NLLBTranslator:
    def __init__(
        self, model_name: str = "facebook/nllb-200-3.3B", device: str = "cuda"
    ):
        """
        Initialize the NLLB translator.

        Args:
            model_name: NLLB model name (default: nllb-200-3.3B)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

        logger.info("Model loaded successfully")

    def translate_batch(
        self, texts: List[str], src_lang: str, trg_lang: str, max_length: int = 512
    ) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of texts to translate
            src_lang: Source language code (e.g., 'eng_Latn')
            trg_lang: Target language code (e.g., 'fra_Latn')
            max_length: Maximum length for generated translations

        Returns:
            List of translated texts
        """
        # Set source language
        self.tokenizer.src_lang = src_lang

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translations
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(trg_lang),
                max_new_tokens=max_length,
                num_beams=5,
                early_stopping=True,
            )

        # Decode translations
        translations = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return translations


def main():
    parser = argparse.ArgumentParser(description="Translate text using NLLB model")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output JSONL file path")
    parser.add_argument(
        "--src_lang", required=True, help="Source language code (ISO639)"
    )
    parser.add_argument(
        "--trg_lang", required=True, help="Target language code (ISO639)"
    )
    parser.add_argument(
        "--model", default="facebook/nllb-200-3.3B", help="NLLB model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for translation"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--device", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Load data
    data = load_jsonl(args.input_file)
    if not data:
        logger.error("No data loaded. Exiting.")
        return

    # Initialize translator
    translator = NLLBTranslator(model_name=args.model, device=args.device)

    # Set nllb codes
    nllb_src_lang = ISO2NLLB[args.src_lang]["code"]
    nllb_trg_lang = ISO2NLLB[args.trg_lang]["code"]

    # Prepare texts for translation
    texts = [item["translation"][args.src_lang] for item in data]

    if not texts:
        logger.error(f"No valid texts found in field '{args.text_field}'")
        return

    logger.info(
        f"Translating {len(texts)} texts from {nllb_src_lang} to {nllb_trg_lang}"
    )

    # Translate in batches
    result = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Translating batches"):
        batch_texts = texts[i : i + args.batch_size]

        # Translate batch
        batch_translations = translator.translate_batch(
            batch_texts, nllb_src_lang, nllb_trg_lang, args.max_length
        )
        batch_translations = [t.strip() for t in batch_translations]

        for t in batch_translations:
            t_item = {"translation": {args.trg_lang: t}}
            result.append(t_item)

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_jsonl(result, args.output_file)
    logger.info("Translation completed successfully!")


if __name__ == "__main__":
    main()
