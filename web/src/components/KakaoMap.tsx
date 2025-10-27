'use client'

import { useEffect } from 'react'
import { useKakaoMap } from '@/hooks/useKakaoMap'
import MapControls from './MapControls'
import type { MarkerData } from '@/types/kakao'
import type { Place, PlaceNearby } from '@/types/api'

interface KakaoMapProps {
  width?: string
  height?: string
  className?: string
  center?: { lat: number; lng: number }
  level?: number
  places?: Place[] | PlaceNearby[]
  onPlaceClick?: (place: Place | PlaceNearby) => void
}

export default function KakaoMap({ 
  width = '100%', 
  height = '452px', 
  className = '',
  center,
  level,
  places = [],
  onPlaceClick
}: KakaoMapProps) {
  const { mapRef, mapInstance, isLoading, error, addMarkers, setLevel } = useKakaoMap({
    center,
    level
  })

  // API 데이터를 기반으로 마커 생성
  useEffect(() => {
    if (!mapInstance || isLoading || !places.length) return

    const placeMarkers: MarkerData[] = places.map((place) => {
      // 장소 타입에 따른 아이콘과 스타일 결정
      const getMarkerInfo = (types: string[]) => {
        if (types.includes('RENT')) {
          return { icon: '☕', style: 'blue-rect' as const, type: 'rent' as const }
        }
        if (types.includes('RETURN')) {
          return { icon: '♻️', style: 'green-circle' as const, type: 'return' as const }
        }
        if (types.includes('BONUS')) {
          return { icon: '🏪', style: 'yellow-circle' as const, type: 'bonus' as const }
        }
        if (types.includes('CLEAN')) {
          return { icon: '🧼', style: 'green-circle' as const, type: 'clean' as const }
        }
        return { icon: '📍', style: 'default' as const, type: 'default' as const }
      }

      const markerInfo = getMarkerInfo(place.types)
      
      return {
        lat: place.location?.lat || 0,
        lng: place.location?.lng || 0,
        title: place.name,
        icon: markerInfo.icon,
        type: markerInfo.type,
        markerStyle: markerInfo.style,
        placeId: place.id,
        onClick: () => onPlaceClick?.(place)
      }
    }).filter(marker => marker.lat !== 0 && marker.lng !== 0) // 유효한 좌표만 필터링

    // 약간의 지연을 두어 맵이 완전히 로드된 후 마커 추가
    const timer = setTimeout(() => {
      addMarkers(placeMarkers)
    }, 100)

    return () => clearTimeout(timer)
  }, [mapInstance, isLoading, places, addMarkers, onPlaceClick])

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

  // 줌 컨트롤 핸들러
  const handleZoomIn = () => {
    if (mapInstance) {
      const currentLevel = mapInstance.getLevel()
      setLevel(currentLevel - 1)
    }
  }

  const handleZoomOut = () => {
    if (mapInstance) {
      const currentLevel = mapInstance.getLevel()
      setLevel(currentLevel + 1)
    }
  }

  const handleCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords
          if (mapInstance) {
            const moveLatLon = new window.kakao.maps.LatLng(latitude, longitude)
            mapInstance.setCenter(moveLatLon)
            setLevel(3)
          }
        },
        (error) => {
          console.error('위치 정보를 가져올 수 없습니다:', error)
        }
      )
    }
  }

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      {/* 맵 컨테이너 */}
      <div 
        ref={mapRef} 
        style={{ width: '100%', height: '100%' }}
        className="rounded-md overflow-hidden bg-gray-100"
      />
      
      {/* 맵 컨트롤 */}
      <MapControls 
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onCurrentLocation={handleCurrentLocation}
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
