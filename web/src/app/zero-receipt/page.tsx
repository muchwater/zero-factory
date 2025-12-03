'use client'

import { useState, useEffect, Suspense } from 'react'
import { useRouter } from 'next/navigation'
import BottomNavigation from '@/components/BottomNavigation'
import { useMember } from '@/hooks/useMember'
import { receiptsApi } from '@/services/api'
import { compressImage } from '@/utils/imageCompression'

// useSearchParams를 사용하던 컴포넌트를 sessionStorage 기반으로 변경
function ZeroReceiptContent() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'profile'>('search')
  const [productDescription, setProductDescription] = useState('')
  const [photo, setPhoto] = useState<File | null>(null)
  const [photoPreview, setPhotoPreview] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const { member, loading, refreshMember } = useMember()

  // 카메라 페이지에서 sessionStorage로 전달된 사진 데이터 복원
  useEffect(() => {
    if (typeof window === 'undefined') return

    const photoData = sessionStorage.getItem('zeroReceiptPhoto')
    if (!photoData) return

    setPhotoPreview(photoData)

    // Data URL을 File 객체로 변환
    fetch(photoData)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' })
        setPhoto(file)
      })
      .catch(err => {
        console.error('failed to restore photo from sessionStorage:', err)
      })
      .finally(() => {
        // 한 번 사용했으면 바로 제거
        sessionStorage.removeItem('zeroReceiptPhoto')
      })
  }, [])

  const handleTabChange = (tab: 'home' | 'search' | 'profile') => {
    setActiveTab(tab)
    if (tab === 'home') {
      router.push('/')
    } else if (tab === 'profile') {
      router.push('/profile')
    }
  }

  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setPhoto(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPhotoPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleCameraClick = () => {
    router.push('/zero-receipt/camera')
  }

  const handleSubmit = async () => {
    if (!member || !photo) {
      alert('회원 정보를 불러오는 중입니다. 잠시 후 다시 시도해주세요.')
      return
    }

    setSubmitting(true)
    try {
      // 이미지 압축 (1200px, 70% 품질)
      const compressedPhoto = await compressImage(photo, 1200, 0.7)

      await receiptsApi.submitReceipt({
        memberId: member.id,
        productDescription,
        photoFile: compressedPhoto,
      })

      // 포인트 잔액 새로고침
      await refreshMember()

      alert(`${member.nickname}님, 영수증이 제출되었습니다! 100포인트가 적립되었습니다.`)

      // 폼 초기화
      setProductDescription('')
      setPhoto(null)
      setPhotoPreview(null)
    } catch (error) {
      console.error('영수증 제출 실패:', error)
      alert('영수증 제출에 실패했습니다. 다시 시도해주세요.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="bg-background min-h-screen flex flex-col pb-20">
      {/* Header */}
      <div className="bg-white border-b border-border sticky top-0 z-40">
        <div className="px-4 py-3">
          <h1 className="text-xl font-bold text-black text-center">제로영수증</h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-4 py-4">
        {/* Section Title */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-foreground">
            당신의 소비를 제보해주세요.
          </h2>
        </div>

        {/* Step 1: Product Description */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            Step 1: 구입한 제품/서비스를 서술해주세요.
          </label>
          <input
            type="text"
            value={productDescription}
            onChange={(e) => setProductDescription(e.target.value)}
            placeholder="E.g. One Americano"
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>

        {/* Step 2: Photo Upload */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            Step 2: 증빙을 위한 사진을 촬영해주세요.
          </label>
          <div className="space-y-2">
            {/* Camera Button */}
            <button
              onClick={handleCameraClick}
              className="block w-full h-32 border-2 border-dashed border-primary rounded-md flex flex-col items-center justify-center cursor-pointer hover:border-primary-dark transition-colors bg-primary/5"
            >
              <svg
                className="w-8 h-8 text-primary mx-auto mb-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
              <p className="text-sm font-medium text-primary">Tap to upload photo</p>
            </button>

            {/* Photo Preview */}
            {photoPreview && (
              <div className="relative">
                <img
                  src={photoPreview}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-md border border-gray-300"
                />
                <button
                  onClick={() => {
                    setPhoto(null)
                    setPhotoPreview(null)
                  }}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-red-600"
                >
                  ×
                </button>
              </div>
            )}

            {/* Alternative: File Upload */}
            <label className="block w-full text-center">
              <input
                type="file"
                accept="image/*"
                onChange={handlePhotoUpload}
                className="hidden"
              />
              <span className="text-xs text-muted underline cursor-pointer">
                또는 파일에서 선택
              </span>
            </label>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-center mt-8">
          {loading ? (
            <div className="text-sm text-muted">회원 정보를 불러오는 중...</div>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!productDescription || !photo || !member || submitting}
              className={`
                px-12 py-3 rounded-xl font-bold text-sm text-white
                transition-all duration-300
                ${productDescription && photo && member && !submitting
                  ? 'bg-primary hover:bg-primary-dark shadow-md hover:shadow-lg'
                  : 'bg-gray-400 cursor-not-allowed'
                }
              `}
            >
              {submitting ? '제출 중...' : '영수증 제출하기'}
            </button>
          )}
        </div>
      </div>

      {/* Bottom Navigation */}
      <BottomNavigation
        activeTab={activeTab}
        onTabChange={handleTabChange}
      />
    </div>
  )
}

// Suspense로 감싼 메인 컴포넌트
export default function ZeroReceiptPage() {
  return (
    <Suspense fallback={
      <div className="bg-background min-h-screen flex flex-col pb-20">
        <div className="bg-white border-b border-border sticky top-0 z-40">
          <div className="px-4 py-3">
            <h1 className="text-xl font-bold text-black text-center">제로영수증</h1>
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-muted">로딩 중...</div>
        </div>
      </div>
    }>
      <ZeroReceiptContent />
    </Suspense>
  )
}
