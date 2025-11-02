'use client'

import { useState, useEffect } from 'react'

interface PlaceFormProps {
  onSubmit: (data: any) => void
  isSubmitting?: boolean
}

interface FormData {
  facilityType: string
  services: string[]
  name: string
  address: string
  detailAddress: string
  contact: string
  opinion: string
  photos: File[]
}

export default function PlaceForm({ onSubmit, isSubmitting = false }: PlaceFormProps) {
  const [formData, setFormData] = useState<FormData>({
    facilityType: '',
    services: [],
    name: '',
    address: '',
    detailAddress: '',
    contact: '',
    opinion: '',
    photos: []
  })

  const [addressSuggestions, setAddressSuggestions] = useState<any[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [coordinates, setCoordinates] = useState<{ lat: number; lng: number } | null>(null)
  const [isKakaoLoaded, setIsKakaoLoaded] = useState(false)

  // Kakao Maps SDK 로드 확인
  useEffect(() => {
    const checkKakaoLoaded = () => {
      if (window.kakao && window.kakao.maps) {
        window.kakao.maps.load(() => {
          setIsKakaoLoaded(true)
          console.log('Kakao Maps SDK 로드 완료')
        })
      } else {
        setTimeout(checkKakaoLoaded, 100)
      }
    }
    checkKakaoLoaded()
  }, [])

  const facilityTypes = [
    { id: 'reusable-container', label: '리유저블 컨테이너', icon: '♻️' },
    { id: 'rvm', label: 'RVM', icon: '🗑️' },
    { id: 'refill-shop', label: '리필샵', icon: '🏪' },
    { id: 'tumbler-cleaner', label: '텀블러 세척기', icon: '🧼' }
  ]

  const serviceOptions = [
    { id: 'rent', label: '대여' },
    { id: 'return', label: '반납' },
    { id: 'bonus', label: '보너스 지급' }
  ]

  const handleInputChange = (field: keyof FormData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handleServiceToggle = (serviceId: string) => {
    setFormData(prev => ({
      ...prev,
      services: prev.services.includes(serviceId)
        ? prev.services.filter(s => s !== serviceId)
        : [...prev.services, serviceId]
    }))
  }

  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setFormData(prev => ({
      ...prev,
      photos: [...prev.photos, ...files]
    }))
  }

  // 카카오 주소 및 키워드 검색 (장소명, 건물명 포함)
  const searchAddress = async (query: string) => {
    if (!query.trim()) {
      setAddressSuggestions([])
      setShowSuggestions(false)
      return
    }

    // Kakao Maps SDK가 로드될 때까지 대기
    if (!isKakaoLoaded) {
      console.log('Kakao Maps SDK 로딩 중...')
      return
    }

    try {
      const results: any[] = []
      
      // 1. 키워드 검색 (장소명, 건물명 검색)
      const places = new window.kakao.maps.services.Places()
      
      await new Promise((resolve) => {
        places.keywordSearch(query, (placeResults: any, placeStatus: any) => {
          if (placeStatus === window.kakao.maps.services.Status.OK) {
            const placesSuggestions = placeResults.map((item: any) => ({
              address_name: item.address_name,
              road_address_name: item.road_address_name || '',
              place_name: item.place_name,
              x: item.x,
              y: item.y,
              type: 'place' // 장소 검색 결과임을 표시
            }))
            results.push(...placesSuggestions)
          }
          resolve(null)
        })
      })
      
      // 2. 주소 검색 (지번/도로명 주소)
      const geocoder = new window.kakao.maps.services.Geocoder()
      
      await new Promise((resolve) => {
        geocoder.addressSearch(query, (addressResults: any, addressStatus: any) => {
          if (addressStatus === window.kakao.maps.services.Status.OK) {
            const addressSuggestions = addressResults.map((item: any) => ({
              address_name: item.address_name,
              road_address_name: item.road_address?.address_name || '',
              x: item.x,
              y: item.y,
              type: 'address' // 주소 검색 결과임을 표시
            }))
            results.push(...addressSuggestions)
          }
          resolve(null)
        })
      })
      
      console.log('검색 결과:', results.length, '건')
      setAddressSuggestions(results)
      setShowSuggestions(results.length > 0)
      
    } catch (error) {
      console.error('검색 오류:', error)
      setAddressSuggestions([])
      setShowSuggestions(false)
    }
  }

  // 주소 선택 시 좌표 설정
  const selectAddress = (suggestion: any) => {
    setFormData(prev => ({
      ...prev,
      address: suggestion.address_name
    }))
    setCoordinates({
      lat: parseFloat(suggestion.y),
      lng: parseFloat(suggestion.x)
    })
    setShowSuggestions(false)
  }

  // 주소 입력 핸들러 (디바운싱 적용)
  const handleAddressInput = (value: string) => {
    setFormData(prev => ({ ...prev, address: value }))
    
    // 디바운싱: 500ms 후에 검색 실행
    const timeoutId = setTimeout(() => {
      searchAddress(value)
    }, 500)
    
    return () => clearTimeout(timeoutId)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.facilityType) {
      alert('시설 종류를 선택해주세요.')
      return
    }
    if (!formData.name) {
      alert('시설명을 입력해주세요.')
      return
    }
    if (!coordinates) {
      alert('주소를 선택해주세요.')
      return
    }
    
    // 좌표 정보를 포함하여 제출
    onSubmit({
      ...formData,
      coordinates
    })
  }

  return (
    <div className="space-y-3">
      {/* 시설 종류 */}
      <div className="space-y-1">
        <p className="text-sm font-medium text-black">시설 종류</p>
        
        {/* 첫 번째 행 */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => handleInputChange('facilityType', 'reusable-container')}
            className={`flex-1 flex items-center gap-2 p-3 rounded-md border transition-colors ${
              formData.facilityType === 'reusable-container'
                ? 'border-black bg-gray-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
              <span className="text-lg">♻️</span>
            </div>
            <span className="text-sm font-extrabold text-black">리유저블 컨테이너</span>
          </button>
          
          <button
            type="button"
            onClick={() => handleInputChange('facilityType', 'rvm')}
            className={`flex-1 flex items-center gap-2 p-3 rounded-md border transition-colors ${
              formData.facilityType === 'rvm'
                ? 'border-black bg-gray-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
              <span className="text-lg">🗑️</span>
            </div>
            <span className="text-sm font-medium text-black">RVM</span>
          </button>
        </div>

        {/* 두 번째 행 */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => handleInputChange('facilityType', 'refill-shop')}
            className={`flex-1 flex items-center gap-2 p-3 rounded-md border transition-colors ${
              formData.facilityType === 'refill-shop'
                ? 'border-black bg-gray-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
              <span className="text-lg">🏪</span>
            </div>
            <span className="text-sm font-medium text-black">리필샵</span>
          </button>
          
          <button
            type="button"
            onClick={() => handleInputChange('facilityType', 'tumbler-cleaner')}
            className={`flex-1 flex items-center gap-2 p-3 rounded-md border transition-colors ${
              formData.facilityType === 'tumbler-cleaner'
                ? 'border-black bg-gray-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
              <span className="text-lg">🧼</span>
            </div>
            <span className="text-sm font-medium text-black">텀블러 세척기</span>
          </button>
        </div>

        {/* 서비스 옵션 */}
        <div className="flex gap-8 px-3 py-2">
          {serviceOptions.map((service) => (
            <div key={service.id} className="flex items-center gap-2">
              <input
                type="checkbox"
                id={service.id}
                checked={formData.services.includes(service.id)}
                onChange={() => handleServiceToggle(service.id)}
                className="w-3.5 h-3.5 border border-black rounded-sm"
              />
              <label htmlFor={service.id} className="text-sm font-medium text-black">
                {service.label}
              </label>
            </div>
          ))}
        </div>
      </div>

      {/* 시설명 */}
      <div className="space-y-1">
        <p className="text-sm font-medium text-black">시설명</p>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => handleInputChange('name', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="|"
        />
      </div>

      {/* 시설 위치 */}
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-black">시설 위치</p>
        
        <div className="relative">
          <div className="flex items-center gap-2 px-3 py-1.5 border border-gray-300 rounded-md">
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <input
              type="text"
              value={formData.address}
              onChange={(e) => handleAddressInput(e.target.value)}
              onFocus={() => setShowSuggestions(addressSuggestions.length > 0)}
              className="flex-1 text-sm placeholder-gray-500 focus:outline-none"
              placeholder="지번, 도로명, 건물명으로 검색"
            />
            <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          
          {/* 주소 자동완성 드롭다운 */}
          {showSuggestions && addressSuggestions.length > 0 && (
            <div className="absolute top-full left-0 right-0 z-50 bg-white border border-gray-300 rounded-md shadow-lg max-h-48 overflow-y-auto mt-1">
              {addressSuggestions.map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => selectAddress(suggestion)}
                  className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 border-b border-gray-100 last:border-b-0"
                >
                  {/* 장소명이 있으면 표시 */}
                  {suggestion.place_name && (
                    <div className="font-bold text-blue-600">{suggestion.place_name}</div>
                  )}
                  <div className={suggestion.place_name ? "text-xs" : "font-medium"}>
                    {suggestion.address_name}
                  </div>
                  {suggestion.road_address_name && (
                    <div className="text-xs text-gray-500">{suggestion.road_address_name}</div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        <input
          type="text"
          value={formData.detailAddress}
          onChange={(e) => handleInputChange('detailAddress', e.target.value)}
          className="w-full px-3 py-2.5 border border-gray-300 rounded-md text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="건물명 등의 상세주소 입력"
        />
      </div>

      {/* 의견 */}
      <div className="space-y-1">
        <p className="text-sm font-medium text-black">의견</p>
        <input
          type="text"
          value={formData.opinion}
          onChange={(e) => handleInputChange('opinion', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="이 장소에 대한 의견을 남겨주세요. (선택)"
        />
      </div>

      {/* 연락처 */}
      <div className="space-y-1">
        <p className="text-sm font-medium text-black">연락처 (선택)</p>
        <input
          type="tel"
          value={formData.contact}
          onChange={(e) => handleInputChange('contact', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="전화번호를 입력해주세요"
        />
      </div>

      {/* 사진 업로드 */}
      <div className="space-y-1">
        <p className="text-sm font-medium text-black">사진 업로드</p>
        <div className="border border-gray-300 rounded-md h-27 flex flex-col items-center justify-center py-2">
          <input
            type="file"
            id="photo-upload"
            multiple
            accept="image/*"
            onChange={handlePhotoUpload}
            className="hidden"
          />
          <label htmlFor="photo-upload" className="cursor-pointer flex flex-col items-center gap-1">
            <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span className="text-sm text-gray-500">Tap to upload photo</span>
          </label>
        </div>
      </div>

      {/* 제출 버튼 */}
      <div className="flex justify-center pt-3">
        <button
          type="submit"
          onClick={handleSubmit}
          disabled={isSubmitting || !formData.facilityType || !formData.name}
          className="w-full max-w-sm bg-gray-500 text-white py-2.5 px-3 rounded-xl text-sm font-bold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
        >
          Submit Spot
        </button>
      </div>
    </div>
  )
}
