'use client'

import { useEffect } from 'react'
import { useKakaoMap } from '@/hooks/useKakaoMap'
import MapControls from './MapControls'
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
  const { mapRef, mapInstance, isLoading, error, addMarkers, setLevel } = useKakaoMap({
    center,
    level
  })

  // 기본 마커들 추가
  useEffect(() => {
    if (!mapInstance || isLoading) return

    const defaultMarkers: MarkerData[] = [
      // 파란색 사각형 마커들 (선화 관련)
      { 
        lat: 37.5665, 
        lng: 126.9780, 
        title: '선화', 
        icon: '선화',
        type: 'seonhwa',
        markerStyle: 'blue-rect'
      },
      { 
        lat: 37.5700, 
        lng: 126.9800, 
        title: '선화', 
        icon: '선화',
        type: 'seonhwa',
        markerStyle: 'blue-rect'
      },
      { 
        lat: 37.5650, 
        lng: 126.9750, 
        title: '선화', 
        icon: '선화',
        type: 'seonhwa',
        markerStyle: 'blue-rect'
      },
      
      // 초록색 원형 마커들 (재활용 관련)
      { 
        lat: 37.5720, 
        lng: 126.9850, 
        title: '재활용센터', 
        icon: '♻️',
        type: 'recycling',
        markerStyle: 'green-circle'
      },
      { 
        lat: 37.5740, 
        lng: 126.9870, 
        title: '재활용센터', 
        icon: '♻️',
        type: 'recycling',
        markerStyle: 'green-circle'
      },
      { 
        lat: 37.5760, 
        lng: 126.9840, 
        title: '재활용센터', 
        icon: '♻️',
        type: 'recycling',
        markerStyle: 'green-circle'
      },
      { 
        lat: 37.5750, 
        lng: 126.9820, 
        title: '재활용센터', 
        icon: '♻️',
        type: 'recycling',
        markerStyle: 'green-circle'
      },
      
      // 노란색 원형 마커들 (스테이션 관련)
      { 
        lat: 37.5600, 
        lng: 126.9700, 
        title: '스테이션', 
        icon: '🌀',
        type: 'station',
        markerStyle: 'yellow-circle'
      },
      { 
        lat: 37.5580, 
        lng: 126.9720, 
        title: '스테이션', 
        icon: '🌀',
        type: 'station',
        markerStyle: 'yellow-circle'
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
