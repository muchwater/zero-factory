'use client'

import { useEffect, useRef, useState } from 'react'
import type { MarkerData, KakaoMapOptions } from '@/types/kakao'

export const useKakaoMap = (options: KakaoMapOptions = {}) => {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<any>(null)
  const markersRef = useRef<any[]>([])  // 생성된 마커들을 추적

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
    const apiKey = process.env.NEXT_PUBLIC_KAKAO_MAP_KEY

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

  // 맵 초기화 (최초 1회만 실행)
  useEffect(() => {
    if (!isScriptLoaded || !mapRef.current || mapInstanceRef.current) return

    const initializeMap = () => {
      try {
        window.kakao.maps.load(() => {
          if (mapRef.current && !mapInstanceRef.current) {
            const mapOptions = {
              center: new window.kakao.maps.LatLng(
                options.center?.lat || 37.5665,
                options.center?.lng || 126.9780
              ),
              level: options.level || 3
            }
            
            const map = new window.kakao.maps.Map(mapRef.current, mapOptions)
            mapInstanceRef.current = map
            setIsLoading(false)
            console.log('카카오맵 초기화 완료')
          }
        })
      } catch (err) {
        setError(`맵 초기화에 실패했습니다: ${err instanceof Error ? err.message : '알 수 없는 오류'}`)
        setIsLoading(false)
      }
    }

    initializeMap()

    // cleanup 함수: 컴포넌트 언마운트 시 맵 정리
    return () => {
      if (mapInstanceRef.current) {
        console.log('카카오맵 정리')
        mapInstanceRef.current = null
      }
    }
  }, [isScriptLoaded])

  // center 변경 시 맵 중심 이동
  useEffect(() => {
    if (mapInstanceRef.current && options.center && window.kakao) {
      const moveLatLon = new window.kakao.maps.LatLng(
        options.center.lat,
        options.center.lng
      )
      mapInstanceRef.current.setCenter(moveLatLon)
    }
  }, [options.center?.lat, options.center?.lng])

  // level 변경 시 줌 레벨 조정
  useEffect(() => {
    if (mapInstanceRef.current && options.level !== undefined) {
      mapInstanceRef.current.setLevel(options.level)
    }
  }, [options.level])

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
      case 'brand-icon':
        return {
          ...baseStyle,
          background: 'transparent',
          border: 'none',
          width: '36px',
          height: '36px',
          padding: '0',
          overflow: 'visible',
          boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
        }
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

  // 기존 마커 모두 제거
  const clearMarkers = () => {
    markersRef.current.forEach((overlay) => {
      overlay.setMap(null)
    })
    markersRef.current = []
  }

  // GPS 위치 마커 (빨간 점)
  const userLocationMarkerRef = useRef<any>(null)

  // userLocation 변경 시 GPS 마커 업데이트
  useEffect(() => {
    // 맵이 완전히 로드될 때까지 대기
    if (!mapInstanceRef.current || !window.kakao || isLoading) {
      return
    }

    const location = options.userLocation

    // 기존 GPS 마커 제거
    if (userLocationMarkerRef.current) {
      userLocationMarkerRef.current.setMap(null)
      userLocationMarkerRef.current = null
    }

    // GPS 위치가 있으면 빨간 점 마커 추가
    if (location) {
      // 약간의 지연을 두어 맵이 완전히 준비된 후 마커 추가
      const timer = setTimeout(() => {
        if (!mapInstanceRef.current || !window.kakao) return

        const markerContent = `
          <div style="
            width: 16px;
            height: 16px;
            background: #FF0000;
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          "></div>
        `

        const overlay = new window.kakao.maps.CustomOverlay({
          position: new window.kakao.maps.LatLng(location.lat, location.lng),
          content: markerContent,
          yAnchor: 0.5,
          xAnchor: 0.5
        })

        overlay.setMap(mapInstanceRef.current)
        userLocationMarkerRef.current = overlay
        console.log('GPS 위치 마커 추가:', location.lat, location.lng)
      }, 200)

      return () => clearTimeout(timer)
    }
  }, [options.userLocation?.lat, options.userLocation?.lng, isScriptLoaded, isLoading])

  // 마커 추가 함수
  const addMarkers = (markers: MarkerData[]) => {
    if (!mapInstanceRef.current || !window.kakao) {
      console.warn('맵이 초기화되지 않았습니다.', { map: !!mapInstanceRef.current, kakao: !!window.kakao })
      return
    }

    // 기존 마커 제거 (GPS 마커는 제외)
    clearMarkers()

    console.log('마커 추가 시작:', markers.length, '개')

    markers.forEach((markerData) => {
      try {
        const markerStyle = getMarkerStyle(markerData)
        const plusIcon = createPlusIcon()

        // 스타일을 CSS 문자열로 변환
        const styleString = Object.entries(markerStyle)
          .map(([key, value]) => `${key.replace(/([A-Z])/g, '-$1').toLowerCase()}: ${value}`)
          .join('; ')

        // 고유 ID 생성
        const markerId = `marker-${markerData.placeId || Math.random()}`

        // 마커 컨텐츠 생성 (이미지 또는 이모지)
        let markerContent = markerData.icon
        if (markerData.imageUrl) {
          markerContent = `<img src="${markerData.imageUrl}" alt="marker" style="width: 100%; height: 100%; object-fit: contain;" />`
        }

        // 커스텀 오버레이 생성
        const overlay = new window.kakao.maps.CustomOverlay({
          position: new window.kakao.maps.LatLng(markerData.lat, markerData.lng),
          content: `
            <div id="${markerId}" style="${styleString}; cursor: pointer;" class="marker-overlay" data-place-id="${markerData.placeId || ''}">
              ${plusIcon}
              ${markerContent}
            </div>
          `,
          yAnchor: 1.2
        })

        overlay.setMap(mapInstanceRef.current)
        markersRef.current.push(overlay)

        // 클릭 이벤트 추가 (카카오맵 이벤트 시스템 사용)
        if (markerData.onClick) {
          const clickHandler = markerData.onClick
          
          // DOM이 렌더링될 때까지 대기 후 이벤트 추가
          const addClickListener = () => {
            const element = document.getElementById(markerId)
            if (element) {
              // 클릭 이벤트 핸들러 (이벤트 버블링 방지)
              const handleClick = (e: MouseEvent) => {
                e.preventDefault()
                e.stopPropagation()
                clickHandler()
              }
              
              // 기존 이벤트 리스너가 있다면 제거
              element.removeEventListener('click', handleClick)
              // 새 이벤트 리스너 추가
              element.addEventListener('click', handleClick)
              
              console.log('마커 클릭 이벤트 추가 완료:', markerData.title)
            } else {
              // DOM이 아직 준비되지 않았으면 재시도
              setTimeout(addClickListener, 100)
            }
          }
          
          // 약간의 지연을 두어 DOM이 완전히 렌더링된 후 이벤트 추가
          setTimeout(addClickListener, 200)
        }

        console.log('마커 생성 완료:', markerData.title, 'at', markerData.lat, markerData.lng)
      } catch (err) {
        console.error('마커 생성 오류:', err)
      }
    })

    console.log('총', markersRef.current.length, '개 마커가 지도에 추가되었습니다.')
  }

  // 맵 중심 이동 함수
  const moveCenter = (lat: number, lng: number) => {
    if (mapInstanceRef.current && window.kakao) {
      const moveLatLon = new window.kakao.maps.LatLng(lat, lng)
      mapInstanceRef.current.setCenter(moveLatLon)
    }
  }

  // 줌 레벨 변경 함수
  const setLevel = (level: number) => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setLevel(level)
    }
  }

  return {
    mapRef,
    mapInstance: mapInstanceRef.current, // 실제 map 객체 반환
    isLoading,
    error,
    addMarkers,
    moveCenter,
    setLevel
  }
}
