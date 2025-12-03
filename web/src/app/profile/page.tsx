'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import BottomNavigation from '@/components/BottomNavigation'
import { useMember } from '@/hooks/useMember'

export default function ProfilePage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'profile'>('profile')
  const { member, loading } = useMember()

  const handleTabChange = (tab: 'home' | 'search' | 'profile') => {
    setActiveTab(tab)
    if (tab === 'home') {
      router.push('/')
    } else if (tab === 'search') {
      router.push('/zero-receipt')
    }
  }

  // 최근 방문 장소 데이터 (실제로는 API에서 가져와야 함)
  const recentVisits = [
    {
      id: 1,
      name: 'Eco-friendly Store',
      icon: '♻️',
      visitedAt: 'Visited Today',
    },
    {
      id: 2,
      name: 'RVM Station',
      icon: '🗑️',
      visitedAt: 'Visited 3 days ago',
    },
    {
      id: 3,
      name: 'Local Refill Shop',
      icon: '🏪',
      visitedAt: 'Visited on July 13th',
    },
  ]

  // 획득한 보상 데이터 (실제로는 API에서 가져와야 함)
  const rewardsClaimed = [
    {
      id: 1,
      name: 'Eco Warrior Badge',
      icon: '🏆',
      claimedAt: 'Claimed on Sep 20th',
    },
    {
      id: 2,
      name: 'Reusable Bag Gift',
      icon: '🎁',
      claimedAt: 'Claimed on Sep 15th',
    },
  ]

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
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-black">내 포인트</p>
            <button
              onClick={() => router.push('/receipts/history')}
              className="text-xs text-primary hover:text-primary-dark font-medium"
            >
              제출 이력 보기 →
            </button>
          </div>
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

        {/* Recent Visits */}
        <div className="space-y-4">
          <div className="pt-4">
            <p className="text-lg font-medium text-black">Recent Visits</p>
          </div>
          <div className="space-y-0">
            {recentVisits.map((visit, index) => (
              <div
                key={visit.id}
                className={`
                  flex gap-2 items-center py-3
                  ${index < recentVisits.length - 1 ? 'border-b border-gray-200' : ''}
                `}
              >
                <div className="w-10 h-10 rounded-2xl bg-gray-100 flex items-center justify-center flex-shrink-0">
                  <span className="text-xl">{visit.icon}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-black">{visit.name}</p>
                  <p className="text-xs text-gray-500">{visit.visitedAt}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Rewards Claimed */}
        <div className="bg-gray-100 rounded-2xl p-3 space-y-4">
          <div className="pt-2">
            <p className="text-lg font-medium text-black">Rewards Claimed</p>
          </div>
          <div className="space-y-0">
            {rewardsClaimed.map((reward, index) => (
              <div
                key={reward.id}
                className={`
                  flex gap-2 items-center py-3
                  ${index < rewardsClaimed.length - 1 ? 'border-b border-gray-300' : ''}
                `}
              >
                <div className="w-8 h-8 rounded-2xl bg-white flex items-center justify-center flex-shrink-0">
                  <span className="text-xl">{reward.icon}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-black">{reward.name}</p>
                  <p className="text-xs text-gray-500">{reward.claimedAt}</p>
                </div>
              </div>
            ))}
          </div>
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

