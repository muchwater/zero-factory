// 카카오맵 API 타입 선언
declare global {
  interface Window {
    kakao: {
      maps: {
        load: (callback: () => void) => void
        LatLng: new (lat: number, lng: number) => any
        LatLngBounds: new () => any
        Map: new (container: HTMLElement, options: any) => any
        Marker: new (options: any) => any
        CustomOverlay: new (options: any) => any
        event: {
          addListener: (target: any, type: string, handler: (...args: any[]) => void) => void
        }
        services: {
          Places: new () => any
          Geocoder: new () => any
          Status: {
            OK: string
            ZERO_RESULT: string
            ERROR: string
          }
        }
      }
    }
  }
}

export interface MarkerData {
  lat: number
  lng: number
  title: string
  icon: string
  imageUrl?: string // 브랜드 아이콘 이미지 URL
  type?: 'cafe' | 'recycling' | 'point' | 'wash' | 'return' | 'seonhwa' | 'station' | 'rent' | 'bonus' | 'clean' | 'default'
  markerStyle?: 'blue-rect' | 'green-circle' | 'yellow-circle' | 'default' | 'brand-icon'
  placeId?: number
  onClick?: () => void
}

export interface KakaoMapOptions {
  center?: {
    lat: number
    lng: number
  }
  level?: number
  userLocation?: {
    lat: number
    lng: number
  }
}

export {}
