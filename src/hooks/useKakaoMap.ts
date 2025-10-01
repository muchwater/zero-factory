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
        console.log('✅ 카카오맵 API 로드 완료')
      } else {
        console.error('❌ 카카오맵 API 로드 후에도 window.kakao.maps가 없습니다')
        setError('카카오맵 API 객체를 찾을 수 없습니다.')
        setIsLoading(false)
      }
    }
    
    script.onerror = (err) => {
      console.error('❌ 카카오맵 API 스크립트 로드 실패:', {
        error: err,
        scriptSrc: script.src,
        readyState: script.readyState,
        apiKey: apiKey ? `${apiKey.substring(0, 8)}...` : 'null'
      })
      
      // 스크립트 URL 직접 테스트
      fetch(script.src)
        .then(response => {
          console.log('직접 fetch 테스트:', {
            status: response.status,
            statusText: response.statusText,
            ok: response.ok
          })
        })
        .catch(fetchErr => {
          console.error('직접 fetch도 실패:', fetchErr)
        })
      
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
            console.log('✅ 카카오맵 초기화 완료:', {
              center: mapOptions.center,
              level: mapOptions.level,
              container: mapRef.current
            })
          }
        })
      } catch (err) {
        console.error('❌ 맵 초기화 오류:', err)
        setError(`맵 초기화에 실패했습니다: ${err instanceof Error ? err.message : '알 수 없는 오류'}`)
        setIsLoading(false)
      }
    }

    initializeMap()
  }, [isScriptLoaded, options.center?.lat, options.center?.lng, options.level])

  // 마커 추가 함수
  const addMarkers = (markers: MarkerData[]) => {
    if (!mapInstance.current || !window.kakao) {
      console.warn('맵이 초기화되지 않았습니다.')
      return
    }

    markers.forEach((markerData) => {
      try {
        // 마커 생성
        const marker = new window.kakao.maps.Marker({
          position: new window.kakao.maps.LatLng(markerData.lat, markerData.lng),
          map: mapInstance.current
        })

        // 커스텀 오버레이 생성
        const overlay = new window.kakao.maps.CustomOverlay({
          position: new window.kakao.maps.LatLng(markerData.lat, markerData.lng),
          content: `
            <div style="
              background: rgba(112,112,112,0.9);
              border-radius: 13px;
              padding: 8px 12px;
              color: white;
              font-size: 14px;
              font-weight: bold;
              text-align: center;
              white-space: nowrap;
              box-shadow: 0 2px 6px rgba(0,0,0,0.3);
              border: 1px solid rgba(255,255,255,0.2);
            ">
              ${markerData.icon} ${markerData.title}
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
