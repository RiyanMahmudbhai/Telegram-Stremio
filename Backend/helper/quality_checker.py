"""
Quality Hierarchy System for comparing video qualities
Prevents accidental replacement of high-quality videos with lower quality

v1.1 - Fixes over v1.0:
  * Token matching is now word-boundary aware. Previously a plain substring
    check caused false positives such as:
      - "384Kbps" matching "4k"   -> scored as 2160p/UHD source
      - "192Kbps" matching "2k"   -> scored as 1440p
      - "Infinity" matching "nf"  -> scored as a Netflix WEB-DL
      - "Camp"     matching "cam"
      - "YTS"      matching "ts"
      - "DVDRip"   matching "dv"  -> scored as Dolby Vision
  * "uhd" removed from the resolution table. In practice "1080p UHD BluRay"
    means a UHD-sourced 1080p encode, not a 2160p file; treating it as 2160p
    made two same-label 1080p files score asymmetrically. It is still counted
    as a top-tier *source*.
  * Separators are normalised (dots/underscores/hyphens -> space) before
    matching, so "web.dl", "web-dl", "WEB DL", "DD5.1" and "DD 5.1" all match
    the same tokens without needing duplicate dictionary keys.
  * Equal-size files are now reported as "identical size" instead of the
    misleading "larger file (370.45MB > 370.45MB)".
"""

import re
from typing import Dict, Optional, Tuple
from Backend.logger import LOGGER


class QualityChecker:
    """
    Checks and compares video quality based on source, resolution, codec, and audio.
    Based on real-world torrent quality patterns.
    """

    # Video Source Quality Rankings (higher = better)
    SOURCE_RANKINGS = {
        # Best Quality
        'bluray': 100, 'blu-ray': 100, 'brrip': 100, 'bdrip': 100,
        'uhd': 100, '4k': 100, 'remux': 100,

        # Excellent Quality
        'web-dl': 85, 'webdl': 85,
        'webrip': 75, 'web-rip': 75,

        # Streaming Platform WEB-DLs (same quality as generic WEB-DL)
        'dsnp': 85, 'nf': 85, 'amzn': 85,  # Disney+, Netflix, Amazon
        'atvp': 85, 'aptv': 85,            # Apple TV+
        'hmax': 85, 'hbo': 85,             # HBO Max

        # Good Quality
        'dvdrip': 60, 'dvd-rip': 60,
        'hdrip': 55, 'hd-rip': 55,

        # Decent Quality
        'hdtv': 50, 'hdtvrip': 50,

        # Lower Quality
        'dvdscr': 40, 'screener': 40,
        'r5': 35,

        # Poor Quality
        'hdcam': 25, 'hd-cam': 25, 'hdts': 25, 'hd-ts': 25,
        'cam': 15, 'camrip': 15, 'cam-rip': 15,
        'ts': 15, 'telesync': 15, 'tc': 15, 'telecine': 15,

        # Worst Quality
        'predvd': 10, 'workprint': 10, 'ppv': 10, 'vhsrip': 5
    }

    # Video Codec Rankings (higher = better)
    CODEC_RANKINGS = {
        'h265': 20, 'hevc': 20, 'x265': 20, 'h.265': 20,
        'av1': 18,
        'h264': 15, 'x264': 15, 'h.264': 15, 'avc': 15,
        'vp9': 12,
        'xvid': 8,
        'divx': 5,
        'mpeg': 3
    }

    # Audio Quality Rankings (higher = better)
    AUDIO_RANKINGS = {
        'atmos': 100, 'dolby atmos': 100,
        'truehd': 95, 'dts-hd': 95, 'dts-hd ma': 95,
        'ddp': 90, 'dd+': 90, 'eac3': 90,
        'dts-x': 88,
        'dts': 85,
        'ddp5.1': 85, 'dd+5.1': 85,
        'dd5.1': 80, 'dd 5.1': 80, 'ac3': 80, 'dolby digital': 80,
        'aac5.1': 65, 'aac 5.1': 65,
        'opus': 60,
        'aac2.0': 50, 'aac 2.0': 50, 'aac': 50,
        'mp3': 40,
        'stereo': 35, '2.0': 35,
        'mono': 20
    }

    # Resolution Rankings (higher = better)
    # NOTE: 'uhd' intentionally NOT here - see module docstring.
    RESOLUTION_RANKINGS = {
        '2160p': 100, '4k': 100,
        '1440p': 80, '2k': 80,
        '1080p': 70, 'fhd': 70,
        '720p': 50,
        '576p': 35,
        '480p': 30,
        '360p': 20,
        '240p': 10
    }

    # HDR/Color Rankings (bonus points)
    HDR_RANKINGS = {
        'hdr10+': 15, 'hdr10plus': 15,
        'hdr10': 12, 'hdr': 12,
        'dolby vision': 18, 'dv': 18,
        'hlg': 10,
        'sdr': 0
    }

    # Sources that imply decent audio even when the filename carries no audio tag
    _BLURAY_SOURCES = ('bluray', 'blu-ray', 'brrip', 'bdrip', 'uhd', '4k', 'remux')
    _WEBDL_SOURCES = ('web-dl', 'webdl', 'dsnp', 'nf', 'amzn', 'atvp', 'aptv', 'hmax', 'hbo')
    _WEBRIP_SOURCES = ('webrip', 'web-rip')

    # Compiled-pattern cache so we build each regex only once
    _pattern_cache: Dict[str, "re.Pattern"] = {}

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Lowercase and collapse separators so that '.', '_', '-' and runs of
        whitespace all become a single space. This lets one dictionary key
        match 'WEB-DL', 'web.dl' and 'WEB DL' alike.
        """
        return re.sub(r'[\s._\-]+', ' ', text.lower()).strip()

    @staticmethod
    def _token_pattern(token: str) -> "re.Pattern":
        """
        Build (and cache) a word-boundary regex for a ranking key.

        Boundaries are 'not alphanumeric' rather than \\b because many tokens
        start or end with digits or symbols ('4k', '2.0', 'dd+', 'hdr10+'),
        where \\b behaves inconsistently. This is what stops '384Kbps' from
        matching '4k' and 'Infinity' from matching 'nf'.
        """
        cached = QualityChecker._pattern_cache.get(token)
        if cached is not None:
            return cached
        normalized = QualityChecker._normalize(token)
        pattern = re.compile(
            r'(?<![a-z0-9])' + re.escape(normalized) + r'(?![a-z0-9])'
        )
        QualityChecker._pattern_cache[token] = pattern
        return pattern

    @staticmethod
    def _best_match(normalized_name: str, rankings: Dict[str, int]) -> Tuple[Optional[str], int]:
        """
        Return the highest-scoring token from `rankings` present in the name.
        """
        best_token, best_score = None, 0
        for token, score in rankings.items():
            if score > best_score and QualityChecker._token_pattern(token).search(normalized_name):
                best_token, best_score = token, score
        return best_token, best_score

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, any]:
        """
        Parse filename to extract quality indicators.
        Returns dict with source, codec, audio, resolution, hdr, etc.
        """
        name = QualityChecker._normalize(filename)

        source, source_score = QualityChecker._best_match(name, QualityChecker.SOURCE_RANKINGS)
        codec, codec_score = QualityChecker._best_match(name, QualityChecker.CODEC_RANKINGS)
        audio, audio_score = QualityChecker._best_match(name, QualityChecker.AUDIO_RANKINGS)
        resolution, resolution_score = QualityChecker._best_match(name, QualityChecker.RESOLUTION_RANKINGS)
        hdr, hdr_score = QualityChecker._best_match(name, QualityChecker.HDR_RANKINGS)

        is_10bit = bool(re.search(r'(?<![a-z0-9])10 ?bit(?![a-z0-9])', name))

        result = {
            'source': source,
            'source_score': source_score,
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
        }

        # Default audio assumption for high-quality sources without explicit audio info.
        # Prevents WEB-DL/BluRay from losing purely because audio isn't in the filename.
        if result['audio_score'] == 0:
            if source in QualityChecker._BLURAY_SOURCES:
                result['audio'] = 'assumed-dd5.1'
                result['audio_score'] = 80   # BluRay: assume standard DD5.1
            elif source in QualityChecker._WEBDL_SOURCES:
                result['audio'] = 'assumed-ddp'
                result['audio_score'] = 85   # Streaming platforms typically ship DD+ (EAC3)
            elif source in QualityChecker._WEBRIP_SOURCES:
                result['audio'] = 'assumed-stereo'
                result['audio_score'] = 45   # Conservative assumption for WEBRip

        return result

    @staticmethod
    def calculate_total_score(parsed_quality: Dict) -> int:
        """
        Calculate total quality score from parsed quality data.
        """
        return (
            parsed_quality['source_score'] +
            parsed_quality['codec_score'] +
            parsed_quality['audio_score'] +
            parsed_quality['resolution_score'] +
            parsed_quality['hdr_score'] +
            parsed_quality['bitrate_bonus']
        )

    @staticmethod
    def parse_file_size(size_str: str) -> float:
        """
        Parse file size string to MB.
        Examples: "2.5GB", "1500MB", "1.5 GB"
        """
        if not size_str:
            return 0.0

        size_str = size_str.upper().replace(' ', '')

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
            parsed['source'] or 'unknown',
            parsed['resolution'] or '',
            parsed['codec'] or '',
            parsed['audio'] or '',
            parsed['hdr'] or '',
        ]
        return ' '.join(p for p in parts if p)

    @staticmethod
    def compare_quality(
        existing_filename: str,
        existing_size: str,
        new_filename: str,
        new_size: str
    ) -> Tuple[bool, str]:
        """
        Compare two video qualities.

        Returns:
            (should_replace: bool, reason: str)

        Logic:
        1. New score > existing        -> REPLACE (genuine upgrade)
        2. Scores equal, new smaller   -> REPLACE (same quality, saves storage)
        3. Scores equal, same/larger   -> SKIP
        4. New score < existing        -> SKIP (protects the better file)
        """
        existing_quality = QualityChecker.parse_filename(existing_filename)
        new_quality = QualityChecker.parse_filename(new_filename)

        existing_score = QualityChecker.calculate_total_score(existing_quality)
        new_score = QualityChecker.calculate_total_score(new_quality)

        existing_size_mb = QualityChecker.parse_file_size(existing_size)
        new_size_mb = QualityChecker.parse_file_size(new_size)

        LOGGER.info("Quality Comparison:")
        LOGGER.info(f"  Existing: {existing_filename}")
        LOGGER.info(f"    Score: {existing_score} (source:{existing_quality['source_score']}, "
                    f"codec:{existing_quality['codec_score']}, audio:{existing_quality['audio_score']}, "
                    f"res:{existing_quality['resolution_score']}, hdr:{existing_quality['hdr_score']}, "
                    f"10bit:{existing_quality['bitrate_bonus']})")
        LOGGER.info(f"    Size: {existing_size} ({existing_size_mb:.2f} MB)")
        LOGGER.info(f"  New: {new_filename}")
        LOGGER.info(f"    Score: {new_score} (source:{new_quality['source_score']}, "
                    f"codec:{new_quality['codec_score']}, audio:{new_quality['audio_score']}, "
                    f"res:{new_quality['resolution_score']}, hdr:{new_quality['hdr_score']}, "
                    f"10bit:{new_quality['bitrate_bonus']})")
        LOGGER.info(f"    Size: {new_size} ({new_size_mb:.2f} MB)")

        if new_score > existing_score:
            reason = f"REPLACE - Better quality (score: {new_score} > {existing_score})"
            LOGGER.info(f"  Decision: {reason}")
            return True, reason

        if new_score == existing_score:
            if new_size_mb > 0 and existing_size_mb > 0:
                if new_size_mb < existing_size_mb:
                    saved = existing_size_mb - new_size_mb
                    reason = (f"REPLACE - Same quality, smaller file "
                              f"({new_size_mb:.2f}MB < {existing_size_mb:.2f}MB, saves {saved:.2f}MB)")
                    LOGGER.info(f"  Decision: {reason}")
                    return True, reason
                if new_size_mb == existing_size_mb:
                    reason = f"SKIP - Same quality and identical size ({new_size_mb:.2f}MB)"
                    LOGGER.info(f"  Decision: {reason}")
                    return False, reason
                reason = (f"SKIP - Same quality, but larger file "
                          f"({new_size_mb:.2f}MB > {existing_size_mb:.2f}MB)")
                LOGGER.info(f"  Decision: {reason}")
                return False, reason

            reason = "REPLACE - Same quality, size unknown"
            LOGGER.info(f"  Decision: {reason}")
            return True, reason

        reason = (f"SKIP - Lower quality detected! (score: {new_score} < {existing_score})\n"
                  f"Existing: {QualityChecker._describe(existing_quality)}\n"
                  f"New: {QualityChecker._describe(new_quality)}")
        LOGGER.warning(f"  Decision: {reason}")
        return False, reason

    @staticmethod
    def should_replace_quality(
        existing_quality_label: str,
        existing_quality_name: str,
        existing_quality_size: str,
        new_quality_label: str,
        new_quality_name: str,
        new_quality_size: str
    ) -> Tuple[bool, str]:
        """
        Main entry point for quality comparison.

        Args:
            existing_quality_label: Resolution label like "1080p", "720p"
            existing_quality_name:  Full filename
            existing_quality_size:  File size string like "2.5GB"
            new_quality_label:      Resolution label
            new_quality_name:       Full filename
            new_quality_size:       File size string

        Returns:
            (should_replace: bool, reason: str)
        """
        # Different resolution labels -> leave the host project's default behaviour alone.
        if existing_quality_label != new_quality_label:
            reason = (f"Different resolution ({existing_quality_label} vs {new_quality_label}) "
                      f"- using default replacement")
            LOGGER.info(reason)
            return True, reason

        return QualityChecker.compare_quality(
            existing_quality_name,
            existing_quality_size,
            new_quality_name,
            new_quality_size
        )
