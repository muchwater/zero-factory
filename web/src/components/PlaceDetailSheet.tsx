'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import type { Place, PlaceNearby } from '@/types/api'
import { calculateDistance, type Coordinates } from '@/utils/location'

interface PlaceDetailSheetProps {
  place: Place | PlaceNearby | null
  onClose: () => void
  userLocation?: Coordinates | null
}

export default function PlaceDetailSheet({ place, onClose, userLocation }: PlaceDetailSheetProps) {
  const router = useRouter()
  const sheetRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const headerRef = useRef<HTMLDivElement>(null)
  const touchStartY = useRef<number>(0)
  const touchStartScrollTop = useRef<number>(0)
  const isDragging = useRef<boolean>(false)
  const currentTranslateY = useRef<number>(0)
  
  // GPS 위치와 가게 위치 간 거리 계산
  const [isWithinRange, setIsWithinRange] = useState<boolean>(false)
  const [dragOffset, setDragOffset] = useState<number>(0)
  
  // place가 변경되면 dragOffset 리셋
  useEffect(() => {
    setDragOffset(0)
    isDragging.current = false
    currentTranslateY.current = 0
  }, [place])
  
  useEffect(() => {
    if (!place || !place.location || !userLocation) {
      setIsWithinRange(false)
      return
    }
    
    const distance = calculateDistance(
      { lat: place.location.lat, lng: place.location.lng },
      { lat: userLocation.lat, lng: userLocation.lng }
    )
    
    // 100m 이내면 활성화
    setIsWithinRange(distance <= 100)
  }, [place, userLocation])

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

  // 스크롤로 닫기 기능
  useEffect(() => {
    const contentElement = contentRef.current
    if (!contentElement) return

    let lastScrollTop = 0
    let isScrollingDown = false

    const handleScroll = () => {
      const currentScrollTop = contentElement.scrollTop
      const scrollHeight = contentElement.scrollHeight
      const clientHeight = contentElement.clientHeight

      // 아래로 스크롤 중인지 확인
      isScrollingDown = currentScrollTop > lastScrollTop
      lastScrollTop = currentScrollTop

      // 맨 아래에 있고, 아래로 스크롤하려고 할 때 닫기
      if (
        currentScrollTop + clientHeight >= scrollHeight - 10 && // 거의 맨 아래
        isScrollingDown
      ) {
        // 약간의 지연을 두어 자연스럽게 닫기
        setTimeout(() => {
          onClose()
        }, 100)
      }
    }

    contentElement.addEventListener('scroll', handleScroll)
    return () => contentElement.removeEventListener('scroll', handleScroll)
  }, [onClose])

  // 헤더 영역 드래그로 닫기
  const handleHeaderTouchStart = (e: React.TouchEvent) => {
    if (contentRef.current && contentRef.current.scrollTop > 0) {
      // 콘텐츠가 스크롤되어 있으면 드래그 비활성화
      return
    }
    isDragging.current = true
    touchStartY.current = e.touches[0].clientY
    currentTranslateY.current = 0
  }

  const handleHeaderTouchMove = (e: React.TouchEvent) => {
    if (!isDragging.current || !sheetRef.current) return

    const currentY = e.touches[0].clientY
    const deltaY = currentY - touchStartY.current

    // 아래로만 드래그 가능
    if (deltaY > 0) {
      currentTranslateY.current = deltaY
      setDragOffset(deltaY)
    }
  }

  const handleHeaderTouchEnd = () => {
    if (!isDragging.current) return

    // 100px 이상 드래그했으면 닫기
    if (currentTranslateY.current > 100) {
      onClose()
    } else {
      // 아니면 원래 위치로 복귀
      setDragOffset(0)
    }

    isDragging.current = false
    currentTranslateY.current = 0
  }

  // 마우스 드래그 지원 (데스크톱)
  const handleHeaderMouseDown = (e: React.MouseEvent) => {
    if (contentRef.current && contentRef.current.scrollTop > 0) {
      return
    }
    isDragging.current = true
    touchStartY.current = e.clientY
    currentTranslateY.current = 0

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !sheetRef.current) return
      const currentY = e.clientY
      const deltaY = currentY - touchStartY.current
      if (deltaY > 0) {
        currentTranslateY.current = deltaY
        setDragOffset(deltaY)
      }
    }

    const handleMouseUp = () => {
      if (!isDragging.current) return
      if (currentTranslateY.current > 100) {
        onClose()
      } else {
        setDragOffset(0)
      }
      isDragging.current = false
      currentTranslateY.current = 0
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }


  // 터치 제스처로 닫기 (모바일 - 하단 스크롤)
  const handleTouchStart = (e: React.TouchEvent) => {
    if (isDragging.current) return
    touchStartY.current = e.touches[0].clientY
    if (contentRef.current) {
      touchStartScrollTop.current = contentRef.current.scrollTop
    }
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!contentRef.current || isDragging.current) return

    const currentY = e.touches[0].clientY
    const deltaY = currentY - touchStartY.current
    const scrollTop = contentRef.current.scrollTop
    const scrollHeight = contentRef.current.scrollHeight
    const clientHeight = contentRef.current.clientHeight

    // 맨 아래에 있고, 아래로 드래그할 때 닫기
    if (
      scrollTop + clientHeight >= scrollHeight - 10 &&
      deltaY > 50
    ) {
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
        className="relative w-full max-w-lg bg-white rounded-t-3xl shadow-2xl animate-slide-up max-h-[90vh]"
        style={{
          animation: dragOffset === 0 ? 'slideUp 0.3s ease-out' : 'none',
          transform: dragOffset > 0 ? `translateY(${dragOffset}px)` : 'none',
          transition: dragOffset === 0 ? 'transform 0.2s ease-out' : 'none',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 드래그 핸들 바 */}
        <div className="flex justify-center pt-3 pb-2">
          <div className="w-10 h-1 bg-gray-300 rounded-full" />
        </div>

        {/* 헤더 (드래그 가능 영역) */}
        <div
          ref={headerRef}
          className="flex items-center justify-between px-2 py-3 cursor-grab active:cursor-grabbing select-none"
          onTouchStart={handleHeaderTouchStart}
          onTouchMove={handleHeaderTouchMove}
          onTouchEnd={handleHeaderTouchEnd}
          onMouseDown={handleHeaderMouseDown}
        >
            <h2 className="text-xl font-medium text-black">{place.name}</h2>
            <div className="flex items-center gap-2">
              <button className="w-6 h-6 flex items-center justify-center text-base">
                🏠
              </button>
              <button className="w-6 h-6 flex items-center justify-center text-base">
                📞
              </button>
              <button
                onClick={onClose}
                className="w-6 h-6 flex items-center justify-center"
                aria-label="닫기"
              >
                <svg
                  className="w-5 h-5 text-gray-600"
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
            </div>
        </div>

        {/* 콘텐츠 */}
        <div
          ref={contentRef}
          className="px-3 pb-3 overflow-y-auto max-h-[90vh]"
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
        >
          {/* 이미지 컨테이너 */}
          <div className="h-[360px] px-3 py-0 relative">
            <div className="w-full h-full bg-gray-100 rounded-md flex items-center justify-center relative">
              <p className="text-base text-black">Main images of the spot</p>
              {/* Pagination */}
              <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 flex gap-1 items-center">
                <div className="w-5 h-1 bg-white rounded-full"></div>
                <div className="w-1 h-1 bg-black/30 rounded-full"></div>
                <div className="w-1 h-1 bg-black/30 rounded-full"></div>
                <div className="w-1 h-1 bg-black/30 rounded-full"></div>
              </div>
            </div>
          </div>

          {/* 매장 이름과 거리 */}
          <div className="px-3 pt-4">
            <p className="text-base font-medium text-black">{place.name}</p>
            {distance !== null && (
              <p className="text-xs text-gray-500 mt-1">
                거리(얼마나 떨어져 있는지)
              </p>
            )}
          </div>

          {/* 액션 버튼 (주소, 영업 시간, 길찾기) */}
          <div className="px-3 pt-4 flex gap-3 justify-center">
            <button className="px-3 py-2.5 border border-black/70 rounded-[13px] text-xs text-black">
              주소
            </button>
            <button className="px-3 py-2.5 border border-black/70 rounded-[13px] text-xs text-black">
              영업 시간
            </button>
            <button
              onClick={() => {
                // 네이버 지도로 길찾기
                if (place.location) {
                  const url = `https://map.naver.com/v5/directions/-/-/-/transit?c=${place.location.lng},${place.location.lat},15,0,0,0,dh`
                  window.open(url, '_blank')
                }
              }}
              className="px-3 py-2.5 border border-black/70 rounded-[13px] text-xs text-black"
            >
              길찾기
            </button>
          </div>

          {/* Services 섹션 */}
          <div className="px-3 pt-4">
            <div className="mb-4">
              <p className="text-lg font-medium text-black">Services</p>
              <p className="text-xs text-gray-500">Services that are provided</p>
            </div>
            <div className="flex gap-3">
              {typeInfos.map((info, index) => (
                <div
                  key={index}
                  className="flex-1 flex flex-col items-center justify-center gap-2 py-3"
                >
                  <div className="w-[50px] h-[50px] bg-gray-100 rounded-2xl flex items-center justify-center">
                    <span className="text-xl">{info.icon}</span>
                  </div>
                  <div className="text-center">
                    <p className="text-xs font-medium text-black leading-tight">
                      {info.label === '다회용컵 대여' ? '메뉴' : info.label}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 제로 영수증 찍기 버튼 */}
          <div className="px-3 pt-4 pb-12 flex justify-center">
            <button
              onClick={() => {
                if (isWithinRange) {
                  onClose()
                  router.push('/zero-receipt')
                }
              }}
              disabled={!isWithinRange}
              className={`w-full max-w-[304px] bg-gray-500 text-white py-2.5 px-3 rounded-[13px] text-sm font-bold ${
                !isWithinRange ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              title={!isWithinRange ? '100m 이내에서만 제로영수증을 사용할 수 있습니다' : ''}
            >
              제로 영수증 찍기
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

