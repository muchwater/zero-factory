'use client'

import { useState } from 'react'

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
    opinion: '',
    photos: []
  })

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
    onSubmit(formData)
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
        
        <div className="flex items-center gap-2 px-3 py-1.5 border border-gray-300 rounded-md">
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <input
            type="text"
            value={formData.address}
            onChange={(e) => handleInputChange('address', e.target.value)}
            className="flex-1 text-sm placeholder-gray-500 focus:outline-none"
            placeholder="지번, 도로명, 건물명으로 검색"
          />
          <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
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
