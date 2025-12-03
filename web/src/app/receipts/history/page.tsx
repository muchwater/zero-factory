'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useMember } from '@/hooks/useMember'
import { receiptsApi } from '@/services/api'
import type { Receipt } from '@/types/api'
import BottomNavigation from '@/components/BottomNavigation'

export default function ReceiptHistoryPage() {
  const router = useRouter()
  const { member, loading: memberLoading } = useMember()
  const [receipts, setReceipts] = useState<Receipt[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'profile'>('profile')

  useEffect(() => {
    if (!member) return

    const loadHistory = async () => {
      setLoading(true)
      try {
        const response = await receiptsApi.getSubmissionHistory({
          memberId: member.id,
          page,
          limit: 10,
        })
        setReceipts(response.receipts)
        setTotalPages(response.totalPages)
      } catch (error) {
        console.error('Failed to load history:', error)
        alert('제출 이력을 불러오는데 실패했습니다.')
      } finally {
        setLoading(false)
      }
    }

    loadHistory()
  }, [member, page])

  const handleTabChange = (tab: 'home' | 'search' | 'profile') => {
    setActiveTab(tab)
    if (tab === 'home') {
      router.push('/')
    } else if (tab === 'search') {
      router.push('/zero-receipt')
    } else if (tab === 'profile') {
      router.push('/profile')
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'APPROVED':
        return (
          <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
            승인
          </span>
        )
      case 'PENDING':
        return (
          <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded">
            대기
          </span>
        )
      case 'REJECTED':
        return (
          <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded">
            거부
          </span>
        )
      default:
        return null
    }
  }

  return (
    <div className="bg-white min-h-screen flex flex-col pb-20">
      {/* Header */}
      <div className="bg-white border-b border-border sticky top-0 z-40 shadow-sm">
        <div className="px-4 py-3 flex items-center">
          <button
            onClick={() => router.back()}
            className="mr-3 text-gray-600"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <h1 className="text-xl font-bold text-black">제출 이력</h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-4 py-4">
        {memberLoading || loading ? (
          <div className="text-center text-gray-500 py-8">
            로딩 중...
          </div>
        ) : receipts.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <p>제출 이력이 없습니다.</p>
            <button
              onClick={() => router.push('/zero-receipt')}
              className="mt-4 px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark"
            >
              영수증 제출하기
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {receipts.map((receipt) => (
              <div
                key={receipt.id}
                className="bg-gray-50 rounded-lg p-4 border border-gray-200"
              >
                <div className="flex gap-4">
                  {/* Photo Thumbnail */}
                  <div className="w-20 h-20 flex-shrink-0 bg-gray-200 rounded overflow-hidden">
                    <img
                      src={`${process.env.NEXT_PUBLIC_API_URL}/${receipt.photoPath}`}
                      alt="영수증"
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-medium text-black truncate">
                        {receipt.productDescription}
                      </div>
                      {getStatusBadge(receipt.status)}
                    </div>

                    <div className="text-sm text-gray-600 mb-1">
                      포인트: +{receipt.pointsEarned}P
                    </div>

                    <div className="text-xs text-gray-500">
                      {new Date(receipt.createdAt).toLocaleString('ko-KR', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center items-center gap-4 pt-4">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className={`px-4 py-2 rounded ${
                    page === 1
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-primary text-white hover:bg-primary-dark'
                  }`}
                >
                  이전
                </button>

                <span className="text-sm text-gray-600">
                  {page} / {totalPages}
                </span>

                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className={`px-4 py-2 rounded ${
                    page === totalPages
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-primary text-white hover:bg-primary-dark'
                  }`}
                >
                  다음
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bottom Navigation */}
      <BottomNavigation
        activeTab={activeTab}
        onTabChange={handleTabChange}
      />
    </div>
  )
}
