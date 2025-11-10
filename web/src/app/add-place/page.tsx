'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import PlaceForm from '@/components/PlaceForm'
import { placesApi } from '@/services/api'

export default function AddPlacePage() {
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (formData: any) => {
    setIsSubmitting(true)
    try {
      // 시설 종류 레이블 및 type 매핑
      const facilityTypeMap: { [key: string]: { label: string; type: string } } = {
        'reusable-container': { label: '리유저블 컨테이너', type: 'RENT' },
        'rvm': { label: 'RVM', type: 'RETURN' },
        'incentive': { label: '인센티브', type: 'BONUS' },
        'tumbler-cleaner': { label: '텀블러 세척기', type: 'CLEAN' }
      }
      
      const facilityInfo = facilityTypeMap[formData.facilityType]
      const facilityTypeLabel = facilityInfo?.label || formData.facilityType
      const facilityType = facilityInfo?.type || 'RENT' // 기본값
      
      // description 생성 (시설 종류 + 의견)
      const description = formData.opinion 
        ? `[${facilityTypeLabel}] ${formData.opinion}`
        : `[${facilityTypeLabel}]`
      
      // API 호출로 장소 데이터 전송
      const newPlace = await placesApi.createPlace({
        name: formData.name,
        address: formData.address,
        detailAddress: formData.detailAddress,
        category: 'FACILITY', // 시설 제보이므로 FACILITY로 고정
        types: [facilityType], // 시설 종류에 따라 자동으로 type 설정
        description: description,
        contact: formData.contact,
        reportedBrand: formData.serviceName, // 서비스명 추가
        coordinates: formData.coordinates
      })
      
      console.log('장소 제보 성공:', newPlace)
      alert('장소가 성공적으로 제보되었습니다!')
      
      // 성공 시 메인 페이지로 이동
      router.push('/')
    } catch (error) {
      console.error('장소 제보 실패:', error)
      alert('장소 제보에 실패했습니다. 다시 시도해주세요.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancel = () => {
    router.push('/')
  }

  return (
    <div className="bg-white min-h-screen flex flex-col">
      {/* Top Bar */}
      <div className="bg-white shadow-[0px_0px_6px_0px_rgba(0,0,0,0.12)] sticky top-0 z-40">
        {/* Status Bar */}
        <div className="h-6 bg-white"></div>
        
        {/* Header */}
        <div className="h-12 flex items-center px-4">
          <button
            onClick={handleCancel}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
          >
            <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <h1 className="flex-1 text-center text-xl font-medium text-black">시설 제보하기</h1>
          <div className="w-6"></div>
        </div>
      </div>

      {/* Form Section */}
      <div className="flex-1 px-3 py-0 pb-16">
        <PlaceForm 
          onSubmit={handleSubmit}
          isSubmitting={isSubmitting}
        />
      </div>
    </div>
  )
}
