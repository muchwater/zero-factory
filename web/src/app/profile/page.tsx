'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import BottomNavigation from '@/components/BottomNavigation'
import { useMember } from '@/hooks/useMember'
import { receiptsApi } from '@/services/api'
import type { Receipt } from '@/types/api'

export default function ProfilePage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'profile'>('profile')
  const { member, loading } = useMember()
  const [receipts, setReceipts] = useState<Receipt[]>([])
  const [receiptsLoading, setReceiptsLoading] = useState(false)
  const [showAllReceipts, setShowAllReceipts] = useState(false)

  // 적립 내역 로드
  useEffect(() => {
    if (!member) return

    const loadReceipts = async () => {
      setReceiptsLoading(true)
      try {
        const response = await receiptsApi.getSubmissionHistory({
          memberId: member.id,
          page: 1,
          limit: 20,
        })
        setReceipts(response.receipts)
      } catch (error) {
        console.error('Failed to load receipts:', error)
      } finally {
        setReceiptsLoading(false)
      }
    }

    loadReceipts()
  }, [member])

  const handleTabChange = (tab: 'home' | 'search' | 'profile') => {
    setActiveTab(tab)
    if (tab === 'home') {
      router.push('/')
    } else if (tab === 'search') {
      router.push('/zero-receipt')
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'APPROVED':
        return (
          <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-700 rounded-full">
            승인
          </span>
        )
      case 'PENDING':
        return (
          <span className="px-2 py-0.5 text-xs font-medium bg-yellow-100 text-yellow-700 rounded-full">
            대기
          </span>
        )
      case 'REJECTED':
        return (
          <span className="px-2 py-0.5 text-xs font-medium bg-red-100 text-red-700 rounded-full">
            거부
          </span>
        )
      default:
        return null
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffTime = Math.abs(now.getTime() - date.getTime())
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24))

    if (diffDays === 0) {
      return '오늘'
    } else if (diffDays === 1) {
      return '어제'
    } else if (diffDays < 7) {
      return `${diffDays}일 전`
    } else {
      return date.toLocaleDateString('ko-KR', {
        month: 'short',
        day: 'numeric',
      })
    }
  }

  // 표시할 적립 내역 (기본 3개, 더보기 시 전체)
  const displayedReceipts = showAllReceipts ? receipts : receipts.slice(0, 3)

  return (
    <div className="bg-white min-h-screen flex flex-col pb-20">
      {/* Header */}
      <div className="bg-white border-b border-border sticky top-0 z-40 shadow-sm">
        <div className="px-4 py-3 flex items-center justify-between">
          <h1 className="text-xl font-bold text-black">ZeroFactory</h1>
          <div className="flex items-center gap-2">
            {loading ? (
              <div className="text-sm text-gray-500">로딩 중...</div>
            ) : member ? (
              <div className="text-sm font-medium text-black">{member.nickname}</div>
            ) : null}
            <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
              <svg
                className="w-5 h-5 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-3 py-4 space-y-4">
        {/* Points Balance */}
        <div className="space-y-1">
          <p className="text-sm font-medium text-black">내 포인트</p>
          <div className="bg-gray-100 rounded-md p-3 flex gap-5 items-center">
            <div className="w-[107px] h-[105px] bg-gradient-to-br from-green-400 to-blue-500 rounded flex flex-col items-center justify-center text-white">
              <div className="text-xs mb-1">POINTS</div>
              {loading ? (
                <div className="text-sm">로딩 중...</div>
              ) : member ? (
                <div className="text-2xl font-bold">{member.pointBalance}</div>
              ) : (
                <div className="text-sm">-</div>
              )}
            </div>
            <div className="flex-1 space-y-1">
              <p className="text-sm text-gray-500">닉네임:</p>
              <p className="text-sm font-medium text-black">
                {loading ? '로딩 중...' : member?.nickname || '-'}
              </p>
              <p className="text-sm text-gray-500">가입일:</p>
              <p className="text-sm font-medium text-black">
                {loading ? '로딩 중...' : member ? new Date(member.createdAt).toLocaleDateString('ko-KR') : '-'}
              </p>
            </div>
          </div>
        </div>

        {/* 적립 내역 */}
        <div className="space-y-3">
          <div className="pt-4 flex items-center justify-between">
            <p className="text-lg font-medium text-black">적립 내역</p>
            {receipts.length > 3 && (
              <button
                onClick={() => setShowAllReceipts(!showAllReceipts)}
                className="text-xs text-primary hover:text-primary-dark font-medium"
              >
                {showAllReceipts ? '접기' : `더보기 (${receipts.length})`}
              </button>
            )}
          </div>

          {receiptsLoading ? (
            <div className="py-8 text-center text-gray-500 text-sm">
              적립 내역을 불러오는 중...
            </div>
          ) : receipts.length === 0 ? (
            <div className="py-8 text-center">
              <div className="text-4xl mb-3">📋</div>
              <p className="text-gray-500 text-sm mb-4">아직 적립 내역이 없습니다.</p>
              <button
                onClick={() => router.push('/zero-receipt')}
                className="px-6 py-2 bg-primary text-white text-sm rounded-lg hover:bg-primary-dark transition-colors"
              >
                첫 적립하러 가기
              </button>
            </div>
          ) : (
            <div className="space-y-0">
              {displayedReceipts.map((receipt, index) => (
                <div
                  key={receipt.id}
                  className={`
                    flex gap-3 items-center py-3
                    ${index < displayedReceipts.length - 1 ? 'border-b border-gray-200' : ''}
                  `}
                >
                  <div className="w-10 h-10 rounded-xl bg-green-100 flex items-center justify-center flex-shrink-0">
                    <span className="text-lg">☕</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="text-sm font-medium text-black truncate">
                        {receipt.productDescription || '다회용기 사용'}
                      </p>
                      {getStatusBadge(receipt.status)}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>{formatDate(receipt.createdAt)}</span>
                      <span>•</span>
                      <span className="text-green-600 font-medium">+{receipt.pointsEarned}P</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 적립하기 버튼 */}
        <div className="pt-4">
          <button
            onClick={() => router.push('/zero-receipt')}
            className="w-full py-3 bg-primary text-white font-bold rounded-xl hover:bg-primary-dark transition-colors flex items-center justify-center gap-2"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            포인트 적립하기
          </button>
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
