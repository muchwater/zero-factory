// API 응답 타입 정의

export interface Place {
  id: number
  name: string
  description?: string
  address: string
  category: 'STORE' | 'FACILITY'
  location?: {
    lat: number
    lng: number
  }
  types: ('RENT' | 'RETURN' | 'BONUS' | 'CLEAN')[]
  contact?: string
  reportedBrand?: string // 제보자가 입력한 서비스명
  brand?: 'SUNHWA' | 'UTURN' // 관리자가 승인 시 선택한 브랜드
  openingHours: StoreOpeningHour[]
  exceptions: StoreOpeningHourException[]
  createdAt: string
  updatedAt: string
}

export interface StoreOpeningHour {
  id: number
  placeId: number
  dayOfWeek: number // 0=일요일 ~ 6=토요일
  isClosed: boolean
  openTime?: string
  closeTime?: string
}

export interface StoreOpeningHourException {
  id: number
  placeId: number
  weekOfMonth?: number
  dayOfWeek?: number
  date?: string
  isClosed: boolean
  openTime?: string
  closeTime?: string
}

export interface PlaceNearby {
  id: number
  name: string
  description?: string
  address: string
  category: 'STORE' | 'FACILITY'
  types: ('RENT' | 'RETURN' | 'BONUS' | 'CLEAN')[]
  contact?: string
  brand?: 'SUNHWA' | 'UTURN' // 관리자가 승인 시 선택한 브랜드
  distance: number // 미터 단위
  location?: {
    lat: number
    lng: number
  }
}

export interface ApiResponse<T> {
  data: T
  message?: string
  status: number
}

export interface Member {
  id: string
  nickname: string
  deviceId: string | null
  pointBalance: number
  lastReceiptAt: string | null  // 마지막 적립 시간
  receiptRestricted: boolean    // 적립 제한 여부
  createdAt: string
}

export interface Receipt {
  id: number
  memberId: string
  placeId?: number
  productDescription: string
  photoPath: string
  pointsEarned: number
  status: 'PENDING' | 'APPROVED' | 'REJECTED'
  verificationResult?: string
  createdAt: string
  updatedAt: string
}

export interface ReceiptHistoryResponse {
  receipts: Receipt[]
  totalCount: number
  currentPage: number
  pageSize: number
  totalPages: number
}

