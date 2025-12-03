// GPS 위치 관련 유틸리티 함수

export interface Coordinates {
  lat: number
  lng: number
}

/**
 * 현재 위치를 Promise로 반환
 */
export function getCurrentPosition(): Promise<Coordinates> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('이 브라우저에서는 위치 서비스를 지원하지 않습니다.'))
      return
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        })
      },
      (error) => {
        switch (error.code) {
          case error.PERMISSION_DENIED:
            reject(new Error('위치 권한이 거부되었습니다. 브라우저 설정에서 위치 권한을 허용해주세요.'))
            break
          case error.POSITION_UNAVAILABLE:
            reject(new Error('위치 정보를 가져올 수 없습니다.'))
            break
          case error.TIMEOUT:
            reject(new Error('위치 정보 요청 시간이 초과되었습니다.'))
            break
          default:
            reject(new Error('알 수 없는 위치 오류가 발생했습니다.'))
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0,
      }
    )
  })
}

/**
 * 두 좌표 간 거리 계산 (Haversine 공식)
 * @returns 거리 (미터 단위)
 */
export function calculateDistance(
  coord1: Coordinates,
  coord2: Coordinates
): number {
  const R = 6371000 // 지구 반경 (미터)
  const dLat = toRadians(coord2.lat - coord1.lat)
  const dLng = toRadians(coord2.lng - coord1.lng)

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(coord1.lat)) *
      Math.cos(toRadians(coord2.lat)) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2)

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  const distance = R * c

  return Math.round(distance) // 소수점 제거
}

function toRadians(degrees: number): number {
  return degrees * (Math.PI / 180)
}

/**
 * 거리를 사용자 친화적인 문자열로 변환
 */
export function formatDistance(meters: number): string {
  if (meters < 1000) {
    return `${meters}m`
  }
  return `${(meters / 1000).toFixed(1)}km`
}

