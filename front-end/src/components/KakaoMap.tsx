'use client'

import { useEffect } from 'react'
import { useKakaoMap } from '@/hooks/useKakaoMap'
import type { MarkerData } from '@/types/kakao'

interface KakaoMapProps {
  width?: string
  height?: string
  className?: string
  center?: { lat: number; lng: number }
  level?: number
}

export default function KakaoMap({ 
  width = '100%', 
  height = '452px', 
  className = '',
  center,
  level 
}: KakaoMapProps) {
  const { mapRef, mapInstance, isLoading, error, addMarkers } = useKakaoMap({
    center,
    level
  })

  // 기본 마커들 추가
  useEffect(() => {
    if (!mapInstance || isLoading) return

    const defaultMarkers: MarkerData[] = [
      { 
        lat: 37.5665, 
        lng: 126.9780, 
        title: '다회용컵 카페', 
        icon: '♻️',
        type: 'cafe'
      },
      { 
        lat: 37.5700, 
        lng: 126.9800, 
        title: '재활용 센터', 
        icon: '♻️',
        type: 'recycling'
      },
      { 
        lat: 37.5600, 
        lng: 126.9700, 
        title: '텀블러 포인트', 
        icon: '🏪',
        type: 'point'
      },
      { 
        lat: 37.5750, 
        lng: 126.9850, 
        title: '세척기', 
        icon: '🧼',
        type: 'wash'
      },
      { 
        lat: 37.5650, 
        lng: 126.9750, 
        title: '반납함', 
        icon: '🗑️',
        type: 'return'
      },
      // 추가 마커들 - 더 다양한 위치
      { 
        lat: 37.5580, 
        lng: 126.9720, 
        title: '에코 카페', 
        icon: '♻️',
        type: 'cafe'
      },
      { 
        lat: 37.5720, 
        lng: 126.9760, 
        title: '친환경 매장', 
        icon: '🏪',
        type: 'point'
      }
    ]

    // 약간의 지연을 두어 맵이 완전히 로드된 후 마커 추가
    const timer = setTimeout(() => {
      addMarkers(defaultMarkers)
    }, 100)

    return () => clearTimeout(timer)
  }, [mapInstance, isLoading, addMarkers])

  if (error) {
    return (
      <div 
        style={{ width, height }}
        className={`rounded-md overflow-hidden bg-gray-100 flex items-center justify-center ${className}`}
      >
        <div className="text-center p-4">
          <div className="text-red-500 text-lg mb-2">⚠️</div>
          <div className="text-sm text-gray-600">
            카카오맵 로드 실패
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {error}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      {/* 맵 컨테이너 */}
      <div 
        ref={mapRef} 
        style={{ width: '100%', height: '100%' }}
        className="rounded-md overflow-hidden bg-gray-100"
      />
      
      {/* 로딩 오버레이 */}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-100 flex items-center justify-center rounded-md">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
            <div className="text-sm text-gray-600">카카오맵 로딩 중...</div>
          </div>
        </div>
      )}
    </div>
  )
}
