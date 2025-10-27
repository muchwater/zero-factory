'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import PlaceForm from '@/components/PlaceForm'

export default function AddPlacePage() {
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (formData: any) => {
    setIsSubmitting(true)
    try {
      // TODO: API 호출로 장소 데이터 전송
      console.log('장소 제보 데이터:', formData)
      
      // 성공 시 메인 페이지로 이동
      router.push('/')
    } catch (error) {
      console.error('장소 제보 실패:', error)
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
