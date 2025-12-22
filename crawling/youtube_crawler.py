"""
유튜브 크롤러
- 키워드: 1_try.txt 기반 (편의점 관련 확장 키워드)
- 수집 컬럼: title, upload_date, view_count, description, url (+채널명 등)
- 연도별 균형 수집: YEARS 설정으로 편향 최소화
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests

# 1_try.txt 기반 키워드
KEYWORDS: Sequence[str] = [
    # ① 외식 및 기호식품 (맛집, 먹방, 트렌드 중심)
    "분식전문점 맛집", "분식전문점 브이로그", "분식전문점 창업 현실", "분식전문점 추천",
    "양식음식점 맛집", "양식음식점 브이로그", "양식음식점 창업 현실", "양식음식점 추천",
    "일식음식점 맛집", "일식음식점 브이로그", "일식음식점 창업 현실", "일식음식점 추천",
    "중식음식점 맛집", "중식음식점 브이로그", "중식음식점 창업 현실", "중식음식점 추천",
    "치킨전문점 맛집", "치킨전문점 브이로그", "치킨전문점 창업 현실", "치킨전문점 추천",
    "커피-음료 맛집", "커피-음료 브이로그", "커피-음료 창업 현실", "커피-음료 추천",
    "패스트푸드점 맛집", "패스트푸드점 브이로그", "패스트푸드점 창업 현실", "패스트푸드점 추천",
    "한식음식점 맛집", "한식음식점 브이로그", "한식음식점 창업 현실", "한식음식점 추천",
    "호프-간이주점 맛집", "호프-간이주점 브이로그", "호프-간이주점 창업 현실", "호프-간이주점 추천",
    "제과점 맛집", "제과점 브이로그", "제과점 빵지순례", "제과점 추천",
    "반찬가게 추천", "반찬가게 브이로그", "반찬가게 반찬 리뷰",

    # ② 서비스 및 여가 (이용 후기, 가격, 시설 중심)
    "PC방 가격", "PC방 이용기", "PC방 근황", "PC방 창업 후기",
    "골프연습장 가격", "골프연습장 이용기", "골프연습장 레슨", "골프연습장 창업 후기",
    "노래방 가격", "노래방 이용기", "노래방 근황", "노래방 창업 후기",
    "당구장 가격", "당구장 이용기", "당구장 근황", "당구장 창업 후기",
    "미용실 가격", "미용실 후기", "미용실 브이로그", "미용실 추천",
    "네일숍 가격", "네일숍 디자인", "네일숍 브이로그", "네일숍 추천",
    "스포츠클럽 이용기", "스포츠클럽 등록", "스포츠클럽 시설",
    "스포츠 강습 후기", "스포츠 강습 가격", "스포츠 강습 브이로그",
    "피부관리실 후기", "피부관리실 효과", "피부관리실 가격",
    "자동차미용 세차", "자동차미용 광택 후기", "자동차미용 가격",
    "자동차수리 정비", "자동차수리 견적", "자동차수리 브이로그",
    "세탁소 이용기", "세탁소 수선 후기", "세탁소 운동화 세탁",
    "고시원 후기", "고시원 브이로그", "고시원 방 구하기",
    "여관 후기", "여관 숙박기", "여관 근황",

    # ③ 소매 및 생활잡화 (하울, 제품 비교, 신상 중심)
    "가구 하울", "가구 추천템", "가구 온라인 비교", "가구 인테리어",
    "가방 하울", "가방 추천템", "가방 왓츠인마이백", "가방 신상",
    "가전제품 하울", "가전제품 추천템", "가전제품 비교 리뷰", "가전제품 신상",
    "문구 하울", "문구 추천템", "문구 다꾸", "문구 신상",
    "신발 하울", "신발 추천템", "신발 사이즈 비교", "신발 신상",
    "완구 하울", "완구 언박싱", "완구 장난감 리뷰",
    "화장품 하울", "화장품 추천템", "화장품 신상 리뷰", "화장품 겟레디윗미",
    "슈퍼마켓 장보기", "슈퍼마켓 하울", "슈퍼마켓 추천템",
    "편의점 하울", "편의점 추천템", "편의점 신상", "편의점 꿀조합",
    "섬유제품 하울", "섬유제품 소재 비교", "섬유제품 추천",
    "운동/경기용품 하울", "운동/경기용품 장비 리뷰", "운동/경기용품 추천",
    "시계및귀금속 하울", "시계및귀금속 브랜드 비교", "시계및귀금속 신상",
    "조명용품 하울", "조명용품 인테리어", "조명용품 추천",
    "화초 하울", "화초 키우기", "화초 인테리어",
    "안경 하울", "안경 브랜드 추천", "안경 얼굴형",
    "반려동물 용품 하울", "애완동물 용품 추천", "애완동물 브이로그",
    "의료기기 리뷰", "의료기기 사용법", "의료기기 부모님 선물",
    "컴퓨터및주변장치판매 하울", "컴퓨터 주변기기 추천", "데스크테리어",
    "전자상거래업 트렌드", "전자상거래업 언박싱", "전자상거래업 하울",

    # ④ 전문 서비스 및 교육 (전망, 정보성, 신뢰 중심)
    "부동산중개업 전망", "부동산중개업 고르는 법", "부동산 사기 안 당하는 법", "부동산중개업 브이로그",
    "외국어학원 전망", "외국어학원 고르는 법", "외국어학원 후기", "외국어학원 브이로그",
    "일반교습학원 전망", "일반교습학원 고르는 법", "학원가 브이로그",
    "예술학원 고르는 법", "예술학원 수강 후기", "예술학원 브이로그",
    "일반의원 전망", "일반의원 고르는 법", "병원 브이로그", "내과 추천",
    "치과의원 고르는 법", "치과 과잉진료 피하는 법", "치과 브이로그",
    "한의원 후기", "한의원 고르는 법", "한의원 브이로그",
    "의약품 정보", "의약품 추천", "약국 꿀템",
    "인테리어 전망", "인테리어 업체 고르는 법", "인테리어 사기 피하기", "인테리어 브이로그",
    "철물점 브이로그", "철물점 꿀템", "철물점 이용법",
    "미곡판매 쌀 고르는 법", "미곡판매 맛있는 쌀 추천",
    "수산물판매 제철 수산물", "수산물판매 고르는 법", "수산시장 브이로그",
    "육류판매 고기 고르는 법", "육류판매 정육점 브이로그", "육류판매 부위별 추천",
    "청과상 과일 고르는 법", "청과상 제철 과일", "청과상 브이로그",
    "핸드폰 싸게 사는 법", "핸드폰 성지", "핸드폰 구매 팁",
    "자전거 및 기타운송장비 추천", "자전거 정비", "입문용 자전거 고르는 법",
    "가전제품수리 팁", "가전제품수리 셀프", "가전제품수리 비용"
]

# 월별 균형 수집 설정
YEARS: Sequence[int] = [2019, 2020, 2021, 2022, 2023, 2024, 2025]  # 수집할 연도들
MONTHS: Sequence[int] = list(range(1, 13))  # 1~12월
MAX_RESULTS_PER_MONTH = 10  # 월별 최대 수집 개수


@dataclass
class VideoItem:
    keyword: str # 어떤 키워드로 검색해서 나온 영상인지
    title: str # 영상 제목
    upload_date: str
    view_count: int # 영상 조회수
    description: str # 영상 설명
    url: str # 영상 url
    source: str  # api 또는 ytdlp
    channel_title: str = "" # 영상 채널명
    like_count: Optional[int] = None # 영상 좋아요 수(API 모드에서만 포함)
    comment_count: Optional[int] = None # 영상 댓글 수(API 모드에서만 포함)
    duration: Optional[str] = None # 영상 길이(API 모드에서만 포함)
    crawled_at: str = datetime.now().isoformat() # 영상 크롤링 일시


class YouTubeAPICrawler:
    def __init__(self, api_key: Optional[str], delay: float = 0.6):
        self.api_key = api_key
        self.delay = delay
        self.base = "https://www.googleapis.com/youtube/v3"

    def _search(self, keyword: str, year: int, month: int, max_results: int) -> List[Dict]:
        """월별 검색 (조회수 높은 순)"""
        if not self.api_key:
            return []
        # 월의 시작일과 다음 달 시작일 계산
        published_after = f"{year}-{month:02d}-01T00:00:00Z"
        if month == 12:
            published_before = f"{year + 1}-01-01T00:00:00Z"
        else:
            published_before = f"{year}-{month + 1:02d}-01T00:00:00Z"
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": max_results,
            "order": "viewCount",  # 조회수 높은 순
            "publishedAfter": published_after,
            "publishedBefore": published_before,
            "key": self.api_key,
        }
        res = requests.get(f"{self.base}/search", params=params, timeout=10)
        res.raise_for_status()
        items = res.json().get("items", [])
        time.sleep(self.delay)
        return items

    def _video_details(self, video_ids: List[str]) -> Dict[str, Dict]:
        if not self.api_key or not video_ids:
            return {}
        params = {
            "part": "statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        res = requests.get(f"{self.base}/videos", params=params, timeout=10)
        res.raise_for_status()
        details = {}
        for item in res.json().get("items", []):
            vid = item["id"]
            stats = item.get("statistics", {})
            details[vid] = {
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)) if "likeCount" in stats else None,
                "comment_count": int(stats.get("commentCount", 0)) if "commentCount" in stats else None,
                "duration": item.get("contentDetails", {}).get("duration"),
            }
        time.sleep(self.delay)
        return details

    def search_keyword_by_months(self, keyword: str, year: int, months: Sequence[int], max_results_per_month: int) -> List[VideoItem]:
        """월별로 조회수 높은 영상 수집"""
        if not self.api_key:
            return []
        results: List[VideoItem] = []
        for month in months:
            try:
                items = self._search(keyword, year, month, max_results_per_month)
                ids = [it["id"]["videoId"] for it in items]
                detail_map = self._video_details(ids)
                for it in items:
                    vid = it["id"]["videoId"]
                    sn = it["snippet"]
                    det = detail_map.get(vid, {})
                    results.append(
                        VideoItem(
                            keyword=keyword,
                            title=sn.get("title", ""),
                            upload_date=sn.get("publishedAt", ""),
                            view_count=det.get("view_count", 0),
                            description=sn.get("description", ""),
                            url=f"https://www.youtube.com/watch?v={vid}",
                            source="api",
                            channel_title=sn.get("channelTitle", ""),
                            like_count=det.get("like_count"),
                            comment_count=det.get("comment_count"),
                            duration=det.get("duration"),
                        )
                    )
                print(f"  └─ {year}년 {month}월: {len(items)}개 수집")
            except Exception as exc:
                print(f"[API] {keyword} {year}년 {month}월 검색 중 오류: {exc}")
                continue
        return results


class YouTubeYtDlpCrawler:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
            self.available = True
        except Exception:
            self.available = False
            print("경고: yt-dlp가 설치되어 있지 않습니다. pip install yt-dlp 후 재시도하세요.")

    def _format_date(self, ymd: Optional[str]) -> str:
        if not ymd:
            return ""
        try:
            return datetime.strptime(ymd, "%Y%m%d").isoformat()
        except Exception:
            return ""

    def search_keyword_by_months(self, keyword: str, year: int, months: Sequence[int], max_results_per_month: int) -> List[VideoItem]:
        """월별로 영상 수집 (yt-dlp는 조회수 정렬 제한적)"""
        if not self.available:
            return []
        results: List[VideoItem] = []
        for month in months:
            try:
                # 월의 시작일과 끝일 계산
                start_date = f"{year}{month:02d}01"
                if month == 12:
                    end_date = f"{year + 1}0101"
                else:
                    end_date = f"{year}{month + 1:02d}01"
                
                query = f"ytsearch{max_results_per_month}:{keyword}"
                cmd = [
                    "yt-dlp",
                    "--dump-json",
                    "--no-download",
                    "--skip-download",
                    "--match-filters",
                    f"upload_date>={start_date} & upload_date<{end_date}",
                    query,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if proc.returncode != 0:
                    print(f"[yt-dlp] {keyword} {year}년 {month}월 실패: {proc.stderr.strip()}")
                    continue
                month_results = []
                for line in proc.stdout.splitlines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    month_results.append(
                        VideoItem(
                            keyword=keyword,
                            title=data.get("title", ""),
                            upload_date=self._format_date(data.get("upload_date")),
                            view_count=int(data.get("view_count", 0) or 0),
                            description=data.get("description", "") or "",
                            url=data.get("webpage_url", ""),
                            source="ytdlp",
                            channel_title=data.get("uploader", ""),
                            duration=str(data.get("duration")) if data.get("duration") else None,
                        )
                    )
                # 조회수 순 정렬 후 상위 N개만 추가
                month_results.sort(key=lambda x: x.view_count, reverse=True)
                results.extend(month_results[:max_results_per_month])
                print(f"  └─ {year}년 {month}월: {len(month_results[:max_results_per_month])}개 수집")
                time.sleep(self.delay)
            except subprocess.TimeoutExpired:
                print(f"[yt-dlp] {keyword} {year}년 {month}월 타임아웃")
            except Exception as exc:
                print(f"[yt-dlp] {keyword} {year}년 {month}월 오류: {exc}")
        return results


def ensure_dirs():
    os.makedirs("youtubeCrawler/results", exist_ok=True)
    os.makedirs("youtubeCrawler/crawling_code", exist_ok=True)


def save_results(items: List[VideoItem], json_path: str, csv_path: str):
    records = [asdict(it) for it in items]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    if records:
        pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")


def main():
    ensure_dirs()
    api_key = os.getenv("YOUTUBE_API_KEY")

    all_items: List[VideoItem] = []

    if api_key:
        print("YouTube Data API 모드로 실행합니다.")
        print(f"수집 대상: {YEARS[0]}~{YEARS[-1]}년, 월별 {MAX_RESULTS_PER_MONTH}개씩 (조회수 순)")
        api_crawler = YouTubeAPICrawler(api_key=api_key, delay=0.6)
        for kw in KEYWORDS:
            print(f"\n[API] '{kw}' 검색 중...")
            for year in YEARS:
                print(f"  {year}년:")
                all_items.extend(api_crawler.search_keyword_by_months(kw, year, MONTHS, MAX_RESULTS_PER_MONTH))
    else:
        print("API 키가 없습니다. yt-dlp 모드로 실행합니다.")
        print(f"수집 대상: {YEARS[0]}~{YEARS[-1]}년, 월별 {MAX_RESULTS_PER_MONTH}개씩")
        ytdlp = YouTubeYtDlpCrawler(delay=1.0)
        for kw in KEYWORDS:
            print(f"\n[yt-dlp] '{kw}' 검색 중...")
            for year in YEARS:
                print(f"  {year}년:")
                all_items.extend(ytdlp.search_keyword_by_months(kw, year, MONTHS, MAX_RESULTS_PER_MONTH))

    # 중복 제거 (같은 URL)
    seen_urls = set()
    unique_items = []
    for item in all_items:
        if item.url not in seen_urls:
            seen_urls.add(item.url)
            unique_items.append(item)
    
    print(f"\n총 수집: {len(all_items)}개 → 중복 제거 후: {len(unique_items)}개")
    save_results(
        unique_items,
        json_path="youtubeCrawler/results/youtube_results.json",
        csv_path="youtubeCrawler/results/youtube_results.csv",
    )
    print("저장 완료: youtubeCrawler/results/")


if __name__ == "__main__":
    main()

