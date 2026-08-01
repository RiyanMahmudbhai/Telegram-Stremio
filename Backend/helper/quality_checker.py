"""
Quality Hierarchy System for comparing video qualities.
Prevents accidental replacement of high-quality videos with lower quality ones.

------------------------------------------------------------------------------
v1.2 changelog (validated against ~100 real-world release names)
------------------------------------------------------------------------------
v1.0 -> v1.1
  * Token matching is now word-boundary aware. A plain `token in filename`
    substring check produced false positives such as:
        "384Kbps"  -> matched "4k"   -> scored as a 2160p/UHD source
        "192Kbps"  -> matched "2k"   -> scored as 1440p
        "Infinity" -> matched "nf"   -> scored as a Netflix WEB-DL
        "DVDRip"   -> matched "dv"   -> scored as Dolby Vision
        "Camp"     -> matched "cam",  "YTS" -> matched "ts"
  * Separators normalised (". _ -" -> space) so "WEB-DL", "web.dl" and
    "WEB DL" all match one dictionary key.
  * Equal-size files report "identical size" instead of the misleading
    "larger file (370.45MB > 370.45MB)".

v1.1 -> v1.2
  * SOURCE IS NOW A HARD HIERARCHY, not just points. Previously a CAM rip with
    a rich audio tag could out-score an untagged disc rip:
        "Tenet.2020.1080p.KOREAN.CAM.H264.AC3"        = 180
        "Tenet IMAX (2020) AC3 5.1 ITA.ENG 1080p H265" = 170  -> CAM won.
    Source tiers (disc > web > tv/rip > screener > cam) are now compared first;
    codec/audio/HDR only break ties inside the same tier. A CAM can never
    replace a WEB-DL or disc rip regardless of its other tags.
  * Resolution now prefers an explicit "NNNNp" token over marketing words.
    "IMAX [BluRay 4K to 1080p ...]" is a 1080p file, not 2160p.
  * Added real-world tokens found missing during testing:
    sources  - bd, bdremux, uhdrip, hdtc, dvdscr variants
    audio    - 5.1 / 7.1 / 6ch / 2ch / 5.1ch style channel-only tags
    codec    - unchanged
  * REMUX now outranks a re-encoded disc rip (untouched stream).
  * Unknown source is treated as "neutral" rather than as the worst tier, so a
    file whose source simply isn't in the filename is not thrown away.
"""

import re
from typing import Dict, Optional, Tuple

from Backend.logger import LOGGER


class QualityChecker:
    """
    Compares two video files by source, resolution, codec, audio and HDR,
    using real-world scene/torrent naming conventions.
    """

    # ---------------------------------------------------------------- sources
    # Score is used for tie-breaking; TIER (below) is what actually decides.
    SOURCE_RANKINGS = {
        # Untouched disc stream - best possible
        'remux': 110, 'bdremux': 110,

        # Disc
        'bluray': 100, 'blu-ray': 100, 'brrip': 100, 'bdrip': 100,
        'bd': 100, 'bd25': 100, 'bd50': 100, 'bdiso': 100,
        'uhd': 100, '4k': 100,

        # Digital Cinema Package (theatrical master) - between web and disc
        'dcprip': 88, 'dcp': 88,

        # Web (untouched stream from a streaming service)
        'web-dl': 85, 'webdl': 85,
        'web': 80,                             # bare "WEB" tag
        # Streaming-platform tags. Scored just under WEB-DL and just over
        # WEBRip: they identify the platform, not the extraction method.
        'dsnp': 78, 'nf': 78, 'amzn': 78,      # Disney+, Netflix, Amazon
        'atvp': 78, 'aptv': 78, 'itunes': 78,  # Apple TV+ / iTunes
        'hmax': 78, 'hbo': 78,                 # HBO Max
        'zee5': 78, 'hotstar': 78, 'sonyliv': 78, 'jio': 78,
        'hulu': 78, 'pcok': 78, 'pmtp': 78, 'stan': 78, 'crav': 78,
        'webrip': 75, 'web-rip': 75, 'webrp': 75,   # 'webrp' = common typo
        'web-dlrip': 70, 'webdlrip': 70,

        # Re-encodes / broadcast
        'dvdrip': 60, 'dvd-rip': 60,
        'uhdrip': 58,
        'hdrip': 55, 'hd-rip': 55, 'vodrip': 55,
        'hdtv': 50, 'hdtvrip': 50,
        'tvrip': 45, 'satrip': 45, 'iptv': 45, 'dthrip': 45,

        # Screeners
        'dvdscr': 40, 'screener': 40, 'scr': 40,
        'r5': 35,

        # Camera / telecine captures
        'hdtc': 28, 'telecine': 28, 'tc': 28,
        'hdcam': 25, 'hd-cam': 25, 'hdts': 25, 'hd-ts': 25,
        'cam': 15, 'camrip': 15, 'cam-rip': 15,
        'ts': 15, 'telesync': 15,

        # Bottom of the barrel
        'predvd': 10, 'pdvd': 10, 'workprint': 10, 'ppv': 10, 'vhsrip': 5,
    }

    # Hard hierarchy. Higher tier ALWAYS beats lower tier, whatever the extras.
    SOURCE_TIERS = {
        4: ('remux', 'bdremux', 'bluray', 'blu-ray', 'brrip', 'bdrip',
            'bd', 'bd25', 'bd50', 'bdiso', 'uhd', '4k'),
        3: ('dcprip', 'dcp',
            'web-dl', 'webdl', 'web',
            'dsnp', 'nf', 'amzn', 'atvp', 'aptv', 'itunes', 'hmax', 'hbo',
            'zee5', 'hotstar', 'sonyliv', 'jio', 'hulu', 'pcok', 'pmtp',
            'stan', 'crav',
            'webrip', 'web-rip', 'webrp', 'web-dlrip', 'webdlrip'),
        2: ('dvdrip', 'dvd-rip', 'uhdrip', 'hdrip', 'hd-rip', 'vodrip',
            'hdtv', 'hdtvrip', 'tvrip', 'satrip', 'iptv', 'dthrip'),
        1: ('dvdscr', 'screener', 'scr', 'r5'),
        0: ('hdtc', 'telecine', 'tc', 'hdcam', 'hd-cam', 'hdts', 'hd-ts',
            'cam', 'camrip', 'cam-rip', 'ts', 'telesync',
            'predvd', 'pdvd', 'workprint', 'ppv', 'vhsrip'),
    }

    # ----------------------------------------------------------------- codecs
    CODEC_RANKINGS = {
        'vvc': 22, 'h266': 22, 'h.266': 22, 'x266': 22,
        'h265': 20, 'hevc': 20, 'x265': 20, 'h.265': 20,
        'av1': 18,
        'h264': 15, 'x264': 15, 'h.264': 15, 'avc': 15,
        'vp9': 12,
        'vc1': 10, 'vc-1': 10,
        'xvid': 8,
        'divx': 5,
        'mpeg': 3, 'mpeg2': 3, 'mpeg-2': 3,
    }

    # ------------------------------------------------------------------ audio
    AUDIO_RANKINGS = {
        'atmos': 100, 'dolby atmos': 100,
        'lpcm': 98, 'pcm': 98,
        'truehd': 95, 'dts-hd': 95, 'dts-hd ma': 95,
        'flac': 92,
        'ddp': 90, 'dd+': 90, 'eac3': 90, 'e-ac3': 90, 'eac-3': 90,
        'dts-x': 88, 'dts:x': 88, 'dts-es': 88,
        'dts': 85,
        'ddp5.1': 85, 'dd+5.1': 85,
        '7.1': 85, '7.1ch': 85, '8ch': 85,
        'dd5.1': 80, 'dd 5.1': 80, 'ac3': 80, 'ac-3': 80, 'dolby digital': 80,
        'aac7.1': 70, 'aac 7.1': 70,
        '5.1': 70, '5.1ch': 70, '6ch': 70,          # channel-only tags
        'aac5.1': 65, 'aac 5.1': 65,
        'opus': 60,
        'vorbis': 55,
        'aac2.0': 50, 'aac 2.0': 50, 'aac': 50,
        'dd2.0': 45, 'dd 2.0': 45,
        'mp3': 40,
        'stereo': 35, '2.0': 35, '2.0ch': 35, '2ch': 35,
        'mono': 20,
    }

    # ------------------------------------------------------------- resolution
    # NOTE: 'uhd' deliberately absent - "1080p UHD BluRay" means a UHD-sourced
    # 1080p encode, not a 2160p file. It still counts as a top-tier *source*.
    RESOLUTION_RANKINGS = {
        '2160p': 100, '4k': 100,
        '1440p': 80, '2k': 80,
        '1080p': 70, 'fhd': 70,
        '720p': 50,
        '576p': 35,
        '480p': 30,
        '360p': 20,
        '240p': 10,
    }

    # An explicit "NNNNp" token always wins over marketing words like "4K".
    _RES_TOKEN_RE = re.compile(r'(?<![a-z0-9])(2160|1440|1080|720|576|480|360|240) ?p(?![a-z0-9])')
    _10BIT_RE = re.compile(r'(?<![a-z0-9])10 ?bits?(?![a-z0-9])')

    # -------------------------------------------------------------------- HDR
    HDR_RANKINGS = {
        'dolby vision': 18, 'dovi': 18, 'dv': 18,
        'hdr10+': 15, 'hdr10plus': 15,
        'hdr10': 12, 'hdr': 12,
        'hlg': 10,
        'sdr': 0,
    }

    # -------------------------------------------------- audio tracks / languages
    # Spoken-language names seen in release titles. Only 3+ letter forms are
    # listed: two-letter ISO codes ("it", "de", "es", "hi", "ja") collide with
    # ordinary words far too often to be safe. Releases that use ISO codes
    # almost always also carry a "Multi" marker, which is handled separately.
    LANGUAGE_TOKENS = (
        'english', 'eng', 'hindi', 'hin', 'bengali', 'bangla', 'ben',
        'tamil', 'tam', 'telugu', 'tel', 'malayalam', 'mal', 'kannada', 'kan',
        'marathi', 'punjabi', 'gujarati', 'urdu', 'nepali',
        'italian', 'ita', 'spanish', 'spa', 'french', 'fre', 'fra', 'vff', 'vfq',
        'german', 'ger', 'deu', 'russian', 'rus', 'ukrainian', 'ukr',
        'korean', 'kor', 'japanese', 'jpn', 'chinese', 'chi', 'mandarin', 'cantonese',
        'portuguese', 'por', 'dutch', 'polish', 'pol', 'turkish', 'tur',
        'arabic', 'ara', 'thai', 'tha', 'vietnamese', 'vie', 'indonesian',
        'tagalog', 'filipino', 'czech', 'hungarian', 'hun', 'swedish', 'swe',
        'danish', 'dan', 'norwegian', 'finnish', 'fin', 'greek', 'gre',
        'hebrew', 'heb', 'persian', 'farsi',
    )

    # Explicit "more than one audio track" markers.
    # 'multi' is rejected when it is really "Multi Subs" (subtitles, not audio).
    _MULTI_AUDIO_RE = re.compile(
        r'(?<![a-z0-9])(?:'
        r'multi(?:\d+)?(?![a-z0-9])(?!\s+subs?(?![a-z0-9]))'
        r'|dual(?:\s*audio)?(?![a-z0-9])'
        r'|\d\s*audios?(?![a-z0-9])'
        r')'
    )

    # Bonus points for carrying more than one audio track. Deliberately small:
    # enough to break a tie between two otherwise identical files, not enough
    # to override a genuine source or resolution difference.
    MULTI_AUDIO_BONUS = 8       # exactly two languages, or a "Dual Audio" tag
    MULTI_AUDIO_BONUS_MANY = 12  # three or more languages, or a "Multi" tag

    # Optional: languages this instance cares about keeping. Empty by default so
    # behaviour is unchanged; set e.g. ('hindi', 'bengali') in a fork to make the
    # checker protect files that carry those tracks.
    PREFERRED_LANGUAGES: tuple = ()
    PREFERRED_LANGUAGE_BONUS = 6

    # Sources that imply decent audio even with no audio tag in the filename
    _BLURAY_SOURCES = ('remux', 'bdremux', 'bluray', 'blu-ray', 'brrip', 'bdrip',
                       'bd', 'bd25', 'bd50', 'bdiso', 'uhd', '4k')
    _WEBDL_SOURCES = ('dcprip', 'dcp', 'web-dl', 'webdl', 'web',
                      'dsnp', 'nf', 'amzn', 'atvp', 'aptv', 'itunes', 'hmax', 'hbo',
                      'zee5', 'hotstar', 'sonyliv', 'jio', 'hulu', 'pcok', 'pmtp',
                      'stan', 'crav')
    _WEBRIP_SOURCES = ('webrip', 'web-rip', 'webrp', 'web-dlrip', 'webdlrip')

    _pattern_cache: Dict[str, "re.Pattern"] = {}

    # ------------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        """
        Lowercase, turn "5,1" into "5.1", then collapse ". _ -" and whitespace
        into single spaces so one dictionary key matches every separator style.
        """
        text = text.lower()
        text = re.sub(r'(?<=\d),(?=\d)', '.', text)
        return re.sub(r'[\s._\-]+', ' ', text).strip()

    @staticmethod
    def _token_pattern(token: str) -> "re.Pattern":
        """
        Word-boundary regex for a ranking key.

        Boundaries are "not alphanumeric" rather than \\b, because many tokens
        begin or end with digits or symbols ('4k', '2.0', 'dd+', 'hdr10+')
        where \\b behaves inconsistently. This is what stops '384Kbps' from
        matching '4k' and 'Infinity' from matching 'nf'.
        """
        cached = QualityChecker._pattern_cache.get(token)
        if cached is not None:
            return cached
        pattern = re.compile(
            r'(?<![a-z0-9])' + re.escape(QualityChecker._normalize(token)) + r'(?![a-z0-9])'
        )
        QualityChecker._pattern_cache[token] = pattern
        return pattern

    @staticmethod
    def _best_match(name: str, rankings: Dict[str, int]) -> Tuple[Optional[str], int]:
        """Highest-scoring token from `rankings` present in the normalised name."""
        best_token, best_score = None, 0
        for token, score in rankings.items():
            if score > best_score and QualityChecker._token_pattern(token).search(name):
                best_token, best_score = token, score
        return best_token, best_score

    @staticmethod
    def source_tier(source: Optional[str]) -> Optional[int]:
        """Hierarchy tier for a source token. None = unknown (neutral)."""
        if not source:
            return None
        for tier, tokens in QualityChecker.SOURCE_TIERS.items():
            if source in tokens:
                return tier
        return None

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, any]:
        """Extract quality indicators from a release filename."""
        name = QualityChecker._normalize(filename)

        source, source_score = QualityChecker._best_match(name, QualityChecker.SOURCE_RANKINGS)
        codec, codec_score = QualityChecker._best_match(name, QualityChecker.CODEC_RANKINGS)
        audio, audio_score = QualityChecker._best_match(name, QualityChecker.AUDIO_RANKINGS)
        hdr, hdr_score = QualityChecker._best_match(name, QualityChecker.HDR_RANKINGS)

        # Resolution: explicit "1080p" style token wins over words like "4K"
        res_match = QualityChecker._RES_TOKEN_RE.search(name)
        if res_match:
            resolution = f"{res_match.group(1)}p"
            resolution_score = QualityChecker.RESOLUTION_RANKINGS.get(resolution, 0)
        else:
            resolution, resolution_score = QualityChecker._best_match(
                name, QualityChecker.RESOLUTION_RANKINGS
            )

        is_10bit = bool(QualityChecker._10BIT_RE.search(name))

        # --- audio tracks / languages -------------------------------------
        languages = sorted(
            lang for lang in QualityChecker.LANGUAGE_TOKENS
            if QualityChecker._token_pattern(lang).search(name)
        )
        has_multi_marker = bool(QualityChecker._MULTI_AUDIO_RE.search(name))

        if len(languages) >= 3 or has_multi_marker:
            multi_audio_bonus = QualityChecker.MULTI_AUDIO_BONUS_MANY
        elif len(languages) == 2:
            multi_audio_bonus = QualityChecker.MULTI_AUDIO_BONUS
        else:
            multi_audio_bonus = 0

        if QualityChecker.PREFERRED_LANGUAGES and any(
            lang in languages for lang in QualityChecker.PREFERRED_LANGUAGES
        ):
            multi_audio_bonus += QualityChecker.PREFERRED_LANGUAGE_BONUS

        result = {
            'source': source,
            'source_score': source_score,
            'source_tier': QualityChecker.source_tier(source),
            'codec': codec,
            'codec_score': codec_score,
            'audio': audio,
            'audio_score': audio_score,
            'resolution': resolution,
            'resolution_score': resolution_score,
            'hdr': hdr,
            'hdr_score': hdr_score,
            'is_10bit': is_10bit,
            'bitrate_bonus': 5 if is_10bit else 0,
            'languages': languages,
            'is_multi_audio': multi_audio_bonus > 0,
            'multi_audio_bonus': multi_audio_bonus,
        }

        # Assume sane audio for good sources that simply carry no audio tag,
        # so a BluRay isn't beaten purely because its name omits "DD5.1".
        if result['audio_score'] == 0:
            if source in QualityChecker._BLURAY_SOURCES:
                result['audio'], result['audio_score'] = 'assumed-dd5.1', 80
            elif source in QualityChecker._WEBDL_SOURCES:
                result['audio'], result['audio_score'] = 'assumed-ddp', 85
            elif source in QualityChecker._WEBRIP_SOURCES:
                result['audio'], result['audio_score'] = 'assumed-stereo', 45

        return result

    @staticmethod
    def calculate_total_score(parsed_quality: Dict) -> int:
        return (
            parsed_quality['source_score']
            + parsed_quality['codec_score']
            + parsed_quality['audio_score']
            + parsed_quality['resolution_score']
            + parsed_quality['hdr_score']
            + parsed_quality['bitrate_bonus']
            + parsed_quality.get('multi_audio_bonus', 0)
        )

    @staticmethod
    def parse_file_size(size_str: str) -> float:
        """Parse a size string ("2.5GB", "1500MB", "1.5 GB") into MB."""
        if not size_str:
            return 0.0

        size_str = str(size_str).upper().replace(' ', '')
        match = re.search(r'([\d.]+)', size_str)
        if not match:
            return 0.0
        try:
            size = float(match.group(1))
        except ValueError:
            return 0.0

        if 'TB' in size_str:
            size *= 1024 * 1024
        elif 'GB' in size_str:
            size *= 1024
        elif 'KB' in size_str:
            size /= 1024
        return size

    @staticmethod
    def _describe(parsed: Dict) -> str:
        parts = [
            parsed['source'] or 'unknown-source',
            parsed['resolution'] or '',
            parsed['codec'] or '',
            parsed['audio'] or '',
            parsed['hdr'] or '',
        ]
        langs = parsed.get('languages') or []
        if langs:
            parts.append('[' + '+'.join(langs) + ']')
        return ' '.join(p for p in parts if p)

    # ------------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------------
    @staticmethod
    def compare_quality(
        existing_filename: str,
        existing_size: str,
        new_filename: str,
        new_size: str,
    ) -> Tuple[bool, str]:
        """
        Decide whether `new` should replace `existing`.

        Order of decision:
          1. Source tier (disc > web > rip/tv > screener > cam). A higher tier
             always wins - a CAM never replaces a WEB-DL no matter how good its
             other tags look, and vice versa.
          2. If one side has no recognisable source, the unknown side is treated
             as neutral EXCEPT that a cam-tier file may never replace it.
          3. Same tier -> total score.
          4. Identical score -> smaller file wins (saves storage).

        Returns (should_replace, reason).
        """
        old = QualityChecker.parse_filename(existing_filename)
        new = QualityChecker.parse_filename(new_filename)

        old_total = QualityChecker.calculate_total_score(old)
        new_total = QualityChecker.calculate_total_score(new)

        old_mb = QualityChecker.parse_file_size(existing_size)
        new_mb = QualityChecker.parse_file_size(new_size)

        LOGGER.info("Quality Comparison:")
        LOGGER.info(f"  Existing: {existing_filename}")
        LOGGER.info(
            f"    Score: {old_total} (source:{old['source_score']}/tier:{old['source_tier']}, "
            f"codec:{old['codec_score']}, audio:{old['audio_score']}, "
            f"res:{old['resolution_score']}, hdr:{old['hdr_score']}, 10bit:{old['bitrate_bonus']}, "
            f"tracks:{old['multi_audio_bonus']} {old['languages'] or '-'})"
        )
        LOGGER.info(f"    Size: {existing_size} ({old_mb:.2f} MB)")
        LOGGER.info(f"  New: {new_filename}")
        LOGGER.info(
            f"    Score: {new_total} (source:{new['source_score']}/tier:{new['source_tier']}, "
            f"codec:{new['codec_score']}, audio:{new['audio_score']}, "
            f"res:{new['resolution_score']}, hdr:{new['hdr_score']}, 10bit:{new['bitrate_bonus']}, "
            f"tracks:{new['multi_audio_bonus']} {new['languages'] or '-'})"
        )
        LOGGER.info(f"    Size: {new_size} ({new_mb:.2f} MB)")

        old_tier, new_tier = old['source_tier'], new['source_tier']

        # --- 1. Hard source hierarchy -------------------------------------
        if old_tier is not None and new_tier is not None and old_tier != new_tier:
            if new_tier > old_tier:
                reason = (f"REPLACE - Better source tier "
                          f"({new['source']} > {old['source']})")
                LOGGER.info(f"  Decision: {reason}")
                return True, reason
            reason = (f"SKIP - Lower source tier! "
                      f"({new['source']} < {old['source']})\n"
                      f"Existing: {QualityChecker._describe(old)}\n"
                      f"New: {QualityChecker._describe(new)}")
            LOGGER.warning(f"  Decision: {reason}")
            return False, reason

        # --- 2. Unknown source on one side: never let a cam-tier file win ---
        if old_tier != new_tier:  # exactly one of them is None here
            if new_tier is not None and new_tier <= 1:
                reason = (f"SKIP - New file is a {new['source']} capture, "
                          f"existing source is unrecognised but not a capture\n"
                          f"Existing: {QualityChecker._describe(old)}\n"
                          f"New: {QualityChecker._describe(new)}")
                LOGGER.warning(f"  Decision: {reason}")
                return False, reason
            if old_tier is not None and old_tier <= 1 and new_tier is None:
                reason = (f"REPLACE - Existing file is a {old['source']} capture, "
                          f"new file is not")
                LOGGER.info(f"  Decision: {reason}")
                return True, reason

        # --- 3. Same tier (or both unknown): total score -------------------
        if new_total > old_total:
            reason = f"REPLACE - Better quality (score: {new_total} > {old_total})"
            LOGGER.info(f"  Decision: {reason}")
            return True, reason

        if new_total == old_total:
            # --- 4. Tie -> never trade audio tracks away for a smaller file --
            # A dropped language track cannot be recovered; disk space can.
            if old['is_multi_audio'] and not new['is_multi_audio']:
                reason = (f"SKIP - Existing file carries more audio tracks "
                          f"({'+'.join(old['languages']) or 'multi'} vs "
                          f"{'+'.join(new['languages']) or 'single'})")
                LOGGER.warning(f"  Decision: {reason}")
                return False, reason

            # --- 5. Otherwise prefer the smaller file -----------------------
            if new_mb > 0 and old_mb > 0:
                if new_mb < old_mb:
                    reason = (f"REPLACE - Same quality, smaller file "
                              f"({new_mb:.2f}MB < {old_mb:.2f}MB, saves {old_mb - new_mb:.2f}MB)")
                    LOGGER.info(f"  Decision: {reason}")
                    return True, reason
                if new_mb == old_mb:
                    reason = f"SKIP - Same quality and identical size ({new_mb:.2f}MB)"
                    LOGGER.info(f"  Decision: {reason}")
                    return False, reason
                reason = (f"SKIP - Same quality, but larger file "
                          f"({new_mb:.2f}MB > {old_mb:.2f}MB)")
                LOGGER.info(f"  Decision: {reason}")
                return False, reason

            reason = "REPLACE - Same quality, size unknown"
            LOGGER.info(f"  Decision: {reason}")
            return True, reason

        reason = (f"SKIP - Lower quality detected! (score: {new_total} < {old_total})\n"
                  f"Existing: {QualityChecker._describe(old)}\n"
                  f"New: {QualityChecker._describe(new)}")
        LOGGER.warning(f"  Decision: {reason}")
        return False, reason

    @staticmethod
    def should_replace_quality(
        existing_quality_label: str,
        existing_quality_name: str,
        existing_quality_size: str,
        new_quality_label: str,
        new_quality_name: str,
        new_quality_size: str,
    ) -> Tuple[bool, str]:
        """
        Public entry point.

        Args:
            existing_quality_label: resolution label such as "1080p"
            existing_quality_name:  full filename
            existing_quality_size:  size string such as "2.5GB"
            new_quality_label / new_quality_name / new_quality_size: same, incoming

        Returns:
            (should_replace, reason)
        """
        # Different resolution labels -> leave the host project's default alone.
        if existing_quality_label != new_quality_label:
            reason = (f"Different resolution ({existing_quality_label} vs {new_quality_label}) "
                      f"- using default replacement")
            LOGGER.info(reason)
            return True, reason

        return QualityChecker.compare_quality(
            existing_quality_name,
            existing_quality_size,
            new_quality_name,
            new_quality_size,
        )
