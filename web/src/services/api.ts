// API 서비스 함수들

import type { Place, PlaceNearby } from '@/types/api'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://43.201.190.116:3000'

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!response.ok) {
      throw new ApiError(response.status, `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    if (error instanceof ApiError) {
      throw error
    }
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

// 장소 관련 API
export const placesApi = {
  // 전체 장소 조회
  getAllPlaces: (): Promise<Place[]> => {
    return fetchApi<Place[]>('/places?state=ACTIVE')
  },

  // 근처 장소 검색
  getPlacesNearby: (params: {
    lat: number
    lng: number
    radius: number
    limit?: number
    offset?: number
    types?: string[]
  }): Promise<PlaceNearby[]> => {
    const searchParams = new URLSearchParams({
      lat: params.lat.toString(),
      lng: params.lng.toString(),
      radius: params.radius.toString(),
      ...(params.limit && { limit: params.limit.toString() }),
      ...(params.offset && { offset: params.offset.toString() }),
      ...(params.types && { types: params.types.join(',') }),
    })

    return fetchApi<PlaceNearby[]>(`/places/nearby?${searchParams}`)
  },

  // 특정 장소 조회
  getPlaceById: (id: number): Promise<Place> => {
    return fetchApi<Place>(`/places/${id}`)
  },

  // 새 장소 제보
  createPlace: (params: {
    name: string
    address: string
    detailAddress?: string
    category: 'STORE' | 'FACILITY'
    types: string[]
    description?: string
    contact?: string
    coordinates: {
      lat: number
      lng: number
    }
  }): Promise<Place> => {
    return fetchApi<Place>('/places', {
      method: 'POST',
      body: JSON.stringify({
        name: params.name,
        address: params.address,
        detailAddress: params.detailAddress,
        category: params.category,
        types: params.types,
        description: params.description,
        contact: params.contact,
        location: params.coordinates,
      }),
    })
  },
}

// 헬스체크 API
export const healthApi = {
  ping: (): Promise<{ message: string }> => {
    return fetchApi<{ message: string }>('/ping')
  },

  health: (): Promise<{
    status: string
    uptime: number
    timestamp: string
    database: string
  }> => {
    return fetchApi('/health')
  },
}

// 어드민 API
export const adminApi = {
  // Pending 장소 목록 조회
  getPendingPlaces: (adminCode: string): Promise<Place[]> => {
    return fetchApi<Place[]>('/admin/places/pending', {
      headers: {
        'x-admin-code': adminCode,
      },
    })
  },

  // 장소 승인 (ACTIVE로 변경)
  activatePlace: (id: number, adminCode: string): Promise<Place> => {
    return fetchApi<Place>(`/admin/places/${id}/activate`, {
      method: 'PUT',
      headers: {
        'x-admin-code': adminCode,
      },
    })
  },

  // 장소 거부 (INACTIVE로 변경)
  rejectPlace: (id: number, adminCode: string): Promise<Place> => {
    return fetchApi<Place>(`/admin/places/${id}/reject`, {
      method: 'PUT',
      headers: {
        'x-admin-code': adminCode,
      },
    })
  },
}

export { ApiError }

