// 카카오맵 API 타입 선언
declare global {
  interface Window {
    kakao: {
      maps: {
        load: (callback: () => void) => void
        LatLng: new (lat: number, lng: number) => any
        Map: new (container: HTMLElement, options: any) => any
        Marker: new (options: any) => any
        CustomOverlay: new (options: any) => any
        event: {
          addListener: (target: any, type: string, handler: (...args: any[]) => void) => void
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
  type?: 'cafe' | 'recycling' | 'point' | 'wash' | 'return'
}

export interface KakaoMapOptions {
  center?: {
    lat: number
    lng: number
  }
  level?: number
}

export {}
