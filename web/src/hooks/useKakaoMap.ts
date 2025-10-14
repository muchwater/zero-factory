'use client'

import { useEffect, useRef, useState } from 'react'
import type { MarkerData, KakaoMapOptions } from '@/types/kakao'

export const useKakaoMap = (options: KakaoMapOptions = {}) => {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isScriptLoaded, setIsScriptLoaded] = useState(false)

  // 스크립트 로드 상태 확인
  useEffect(() => {
    const checkScript = () => {
      if (window.kakao && window.kakao.maps) {
        setIsScriptLoaded(true)
        return true
      }
      return false
    }

    // 이미 로드되어 있는지 확인
    if (checkScript()) {
      return
    }

    // 기존 스크립트 태그가 있는지 확인
    const existingScript = document.querySelector('script[src*="dapi.kakao.com"]')
    if (existingScript) {
      existingScript.addEventListener('load', () => {
        checkScript()
      })
      return
    }

    // 새로운 스크립트 태그 생성
    const script = document.createElement('script')
    const apiKey = process.env.NEXT_PUBLIC_KAKAO_MAP_API_KEY
    
    if (!apiKey) {
      setError('카카오맵 API 키가 설정되지 않았습니다.')
      setIsLoading(false)
      return
    }

    script.src = `//dapi.kakao.com/v2/maps/sdk.js?appkey=${apiKey}&autoload=false`
    script.async = true
    
    script.onload = () => {
      if (checkScript()) {
        // API 로드 완료
      } else {
        setError('카카오맵 API 객체를 찾을 수 없습니다.')
        setIsLoading(false)
      }
    }
    
    script.onerror = () => {
      setError('카카오맵 API 스크립트 로드에 실패했습니다. 네트워크 연결을 확인해주세요.')
      setIsLoading(false)
    }
    
    document.head.appendChild(script)
  }, [])

  // 맵 초기화
  useEffect(() => {
    if (!isScriptLoaded || !mapRef.current) return

    const initializeMap = () => {
      try {
        window.kakao.maps.load(() => {
          if (mapRef.current) {
            const mapOptions = {
              center: new window.kakao.maps.LatLng(
                options.center?.lat || 37.5665,
                options.center?.lng || 126.9780
              ),
              level: options.level || 3
            }
            
            mapInstance.current = new window.kakao.maps.Map(mapRef.current, mapOptions)
            setIsLoading(false)
          }
        })
      } catch (err) {
        setError(`맵 초기화에 실패했습니다: ${err instanceof Error ? err.message : '알 수 없는 오류'}`)
        setIsLoading(false)
      }
    }

    initializeMap()
  }, [isScriptLoaded, options.center?.lat, options.center?.lng, options.level])

  // 마커 스타일 생성 함수
  const getMarkerStyle = (markerData: MarkerData) => {
    const baseStyle = {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: 'bold',
      textAlign: 'center',
      whiteSpace: 'nowrap',
      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
      position: 'relative',
      border: '2px solid white'
    }

    switch (markerData.markerStyle) {
      case 'blue-rect':
        return {
          ...baseStyle,
          background: '#4285F4',
          borderRadius: '12px',
          padding: '8px 12px',
          color: 'white',
          fontSize: '13px',
          minWidth: '50px',
          height: '28px',
          fontFamily: 'Arial, sans-serif'
        }
      case 'green-circle':
        return {
          ...baseStyle,
          background: '#34A853',
          borderRadius: '50%',
          width: '36px',
          height: '36px',
          color: 'white',
          fontSize: '18px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }
      case 'yellow-circle':
        return {
          ...baseStyle,
          background: '#FBBC04',
          borderRadius: '50%',
          width: '36px',
          height: '36px',
          color: '#1a73e8',
          fontSize: '18px',
          fontWeight: '900',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }
      default:
        return {
          ...baseStyle,
          background: 'rgba(112,112,112,0.9)',
          borderRadius: '13px',
          padding: '8px 12px',
          color: 'white',
          fontSize: '14px',
          border: '1px solid rgba(255,255,255,0.2)'
        }
    }
  }

  // 플러스 아이콘 생성 함수
  const createPlusIcon = () => {
    return `
      <div style="
        position: absolute;
        top: -4px;
        right: -4px;
        width: 14px;
        height: 14px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        color: #333;
        border: 1px solid #ddd;
        font-weight: bold;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
      ">+</div>
    `
  }

  // 마커 추가 함수
  const addMarkers = (markers: MarkerData[]) => {
    if (!mapInstance.current || !window.kakao) {
      console.warn('맵이 초기화되지 않았습니다.')
      return
    }

    markers.forEach((markerData) => {
      try {
        const markerStyle = getMarkerStyle(markerData)
        const plusIcon = createPlusIcon()
        
        // 스타일을 CSS 문자열로 변환
        const styleString = Object.entries(markerStyle)
          .map(([key, value]) => `${key.replace(/([A-Z])/g, '-$1').toLowerCase()}: ${value}`)
          .join('; ')

        // 커스텀 오버레이 생성
        const overlay = new window.kakao.maps.CustomOverlay({
          position: new window.kakao.maps.LatLng(markerData.lat, markerData.lng),
          content: `
            <div style="${styleString}">
              ${plusIcon}
              ${markerData.icon}
            </div>
          `,
          yAnchor: 1.2
        })

        overlay.setMap(mapInstance.current)
      } catch (err) {
        console.error('마커 생성 오류:', err)
      }
    })
  }

  // 맵 중심 이동 함수
  const moveCenter = (lat: number, lng: number) => {
    if (mapInstance.current && window.kakao) {
      const moveLatLon = new window.kakao.maps.LatLng(lat, lng)
      mapInstance.current.setCenter(moveLatLon)
    }
  }

  // 줌 레벨 변경 함수
  const setLevel = (level: number) => {
    if (mapInstance.current) {
      mapInstance.current.setLevel(level)
    }
  }

  return {
    mapRef,
    mapInstance: mapInstance.current,
    isLoading,
    error,
    addMarkers,
    moveCenter,
    setLevel
  }
}
