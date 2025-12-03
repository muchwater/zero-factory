// API 서비스 함수들

import type { Place, PlaceNearby, Member, Receipt, ReceiptHistoryResponse } from '@/types/api'

// Validate that NEXT_PUBLIC_API_URL is set at build time
if (!process.env.NEXT_PUBLIC_API_URL) {
  throw new Error(
    'NEXT_PUBLIC_API_URL environment variable is not set. ' +
    'Please set it in your .env file before building.'
  )
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL

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
    reportedBrand?: string
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
        reportedBrand: params.reportedBrand,
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

// 회원 (어드민용) 타입
export interface AdminMember {
  id: string
  nickname: string
  pointBalance: number
  lastReceiptAt: string | null
  receiptRestricted: boolean
  createdAt: string
  _count: {
    receipts: number
  }
  receipts3Days: number  // 최근 3일 적립 수 (이상 적립 감지용)
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
  activatePlace: (id: number, adminCode: string, brand?: string): Promise<Place> => {
    const body: { brand?: string } = {}
    if (brand) {
      body.brand = brand
    }
    
    return fetchApi<Place>(`/admin/places/${id}/activate`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'x-admin-code': adminCode,
      },
      body: JSON.stringify(body),
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

  // 전체 회원 목록 조회
  getAllMembers: (adminCode: string): Promise<AdminMember[]> => {
    return fetchApi<AdminMember[]>('/admin/members', {
      headers: {
        'x-admin-code': adminCode,
      },
    })
  },

  // 회원 적립 제한 설정
  restrictMember: (memberId: string, adminCode: string): Promise<AdminMember> => {
    return fetchApi<AdminMember>(`/admin/members/${memberId}/restrict`, {
      method: 'PUT',
      headers: {
        'x-admin-code': adminCode,
      },
    })
  },

  // 회원 적립 제한 해제
  unrestrictMember: (memberId: string, adminCode: string): Promise<AdminMember> => {
    return fetchApi<AdminMember>(`/admin/members/${memberId}/unrestrict`, {
      method: 'PUT',
      headers: {
        'x-admin-code': adminCode,
      },
    })
  },
}

// 회원 관련 API
export const membersApi = {
  // 회원 생성 또는 조회
  findOrCreate: (params: {
    nickname: string
    deviceId?: string
  }): Promise<Member> => {
    return fetchApi<Member>('/members', {
      method: 'POST',
      body: JSON.stringify({
        nickname: params.nickname,
        deviceId: params.deviceId,
      }),
    })
  },

  // 회원 ID로 조회
  getById: (memberId: string): Promise<Member> => {
    return fetchApi<Member>(`/members/${memberId}`)
  },
}

// 영수증 관련 API
export const receiptsApi = {
  // 영수증 제출
  submitReceipt: async (params: {
    memberId: string
    productDescription: string
    photoFile: File
    verificationResult?: any
    placeId?: number
  }): Promise<Receipt> => {
    const formData = new FormData()
    formData.append('photo', params.photoFile)
    formData.append('productDescription', params.productDescription)
    if (params.verificationResult) {
      formData.append('verificationResult', JSON.stringify(params.verificationResult))
    }
    if (params.placeId) {
      formData.append('placeId', params.placeId.toString())
    }

    const response = await fetch(`${API_BASE_URL}/members/${params.memberId}/receipts`, {
      method: 'POST',
      body: formData, // FormData는 Content-Type 자동 설정
    })

    if (!response.ok) {
      throw new ApiError(response.status, `HTTP error! status: ${response.status}`)
    }

    return response.json()
  },

  // 제출 이력 조회
  getSubmissionHistory: (params: {
    memberId: string
    page?: number
    limit?: number
    status?: 'PENDING' | 'APPROVED' | 'REJECTED'
  }): Promise<ReceiptHistoryResponse> => {
    const searchParams = new URLSearchParams({
      page: (params.page || 1).toString(),
      limit: (params.limit || 20).toString(),
      ...(params.status && { status: params.status }),
    })

    return fetchApi<ReceiptHistoryResponse>(
      `/members/${params.memberId}/receipts?${searchParams}`
    )
  },

  // 영수증 상세 조회
  getReceiptById: (receiptId: number): Promise<Receipt> => {
    return fetchApi<Receipt>(`/receipts/${receiptId}`)
  },
}

export { ApiError }




