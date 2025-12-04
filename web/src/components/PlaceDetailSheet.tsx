'use client'

import { useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import type { Place, PlaceNearby } from '@/types/api'

interface PlaceDetailSheetProps {
  place: Place | PlaceNearby | null
  onClose: () => void
}

export default function PlaceDetailSheet({ place, onClose }: PlaceDetailSheetProps) {
  const router = useRouter()
  const sheetRef = useRef<HTMLDivElement>(null)

  // ESC 키로 닫기
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [onClose])

  // 배경 클릭 시 닫기
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  if (!place) return null

  // 타입에 따른 아이콘 및 라벨
  const getTypeInfo = (types: string[]) => {
    const typeMap: Record<string, { icon: string; label: string; color: string }> = {
      RENT: { icon: '☕', label: '다회용컵 대여', color: 'bg-blue-100 text-blue-700' },
      RETURN: { icon: '♻️', label: '반납함', color: 'bg-green-100 text-green-700' },
      BONUS: { icon: '🏪', label: '포인트 적립', color: 'bg-yellow-100 text-yellow-700' },
      CLEAN: { icon: '🧼', label: '세척기', color: 'bg-purple-100 text-purple-700' },
    }

    return types.map((type) => typeMap[type]).filter(Boolean)
  }

  // 브랜드 정보
  const getBrandInfo = (brand?: string) => {
    const brandMap: Record<string, { name: string; color: string }> = {
      SUNHWA: { name: '순환경제', color: 'bg-emerald-100 text-emerald-700' },
      UTURN: { name: '유턴', color: 'bg-sky-100 text-sky-700' },
    }
    return brand ? brandMap[brand] : null
  }

  const typeInfos = getTypeInfo(place.types)
  const brandInfo = getBrandInfo(place.brand)
  const distance = 'distance' in place ? place.distance : null

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center"
      onClick={handleBackdropClick}
    >
      {/* 배경 오버레이 */}
      <div className="absolute inset-0 bg-black/40 transition-opacity" />

      {/* 바텀시트 */}
      <div
        ref={sheetRef}
        className="relative w-full max-w-lg bg-white rounded-t-3xl shadow-2xl animate-slide-up"
        style={{
          animation: 'slideUp 0.3s ease-out',
        }}
      >
        {/* 핸들 바 */}
        <div className="flex justify-center pt-3 pb-2">
          <div className="w-10 h-1 bg-gray-300 rounded-full" />
        </div>

        {/* 닫기 버튼 */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
          aria-label="닫기"
        >
          <svg
            className="w-4 h-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        {/* 콘텐츠 */}
        <div className="px-5 pb-8">
          {/* 장소명 및 브랜드 */}
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <h2 className="text-xl font-bold text-gray-900">{place.name}</h2>
              {brandInfo && (
                <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${brandInfo.color}`}>
                  {brandInfo.name}
                </span>
              )}
            </div>

            {/* 거리 정보 */}
            {distance !== null && (
              <p className="text-sm text-primary font-medium">
                📍 {distance < 1000 ? `${distance}m` : `${(distance / 1000).toFixed(1)}km`} 거리
              </p>
            )}
          </div>

          {/* 타입 태그 */}
          <div className="flex flex-wrap gap-2 mb-4">
            {typeInfos.map((info, index) => (
              <span
                key={index}
                className={`inline-flex items-center gap-1 px-3 py-1.5 text-sm font-medium rounded-full ${info.color}`}
              >
                <span>{info.icon}</span>
                <span>{info.label}</span>
              </span>
            ))}
          </div>

          {/* 주소 */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-500 mb-1">주소</h3>
            <p className="text-gray-900">{place.address}</p>
          </div>

          {/* 설명 */}
          {place.description && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-500 mb-1">설명</h3>
              <p className="text-gray-700 text-sm">{place.description}</p>
            </div>
          )}

          {/* 연락처 */}
          {place.contact && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-500 mb-1">연락처</h3>
              <a
                href={`tel:${place.contact}`}
                className="text-primary hover:underline"
              >
                {place.contact}
              </a>
            </div>
          )}

          {/* 액션 버튼 */}
          <div className="mt-6 flex gap-3">
            <button
              onClick={() => {
                // 네이버 지도로 길찾기
                if (place.location) {
                  const url = `https://map.naver.com/v5/directions/-/-/-/transit?c=${place.location.lng},${place.location.lat},15,0,0,0,dh`
                  window.open(url, '_blank')
                }
              }}
              className="flex-1 py-3 bg-gray-100 text-gray-700 font-medium rounded-xl hover:bg-gray-200 transition-colors flex items-center justify-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
                />
              </svg>
              길찾기
            </button>
            <button
              onClick={() => {
                onClose()
                router.push('/zero-receipt')
              }}
              className="flex-1 py-3 bg-primary text-white font-medium rounded-xl hover:bg-primary-dark transition-colors flex items-center justify-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              제로영수증
            </button>
            <button
              onClick={onClose}
              className="flex-1 py-3 bg-gray-100 text-gray-700 font-medium rounded-xl hover:bg-gray-200 transition-colors"
            >
              닫기
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes slideUp {
          from {
            transform: translateY(100%);
          }
          to {
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  )
}

