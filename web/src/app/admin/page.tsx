'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { adminApi, ApiError, AdminMember } from '@/services/api'
import type { Place } from '@/types/api'

export default function AdminPage() {
  const router = useRouter()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [adminCode, setAdminCode] = useState('')
  const [inputCode, setInputCode] = useState('')
  const [pendingPlaces, setPendingPlaces] = useState<Place[]>([])
  const [members, setMembers] = useState<AdminMember[]>([])
  const [loading, setLoading] = useState(false)
  const [membersLoading, setMembersLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [authError, setAuthError] = useState<string | null>(null)
  const [selectedBrands, setSelectedBrands] = useState<{ [key: number]: string }>({})
  const [activeTab, setActiveTab] = useState<'places' | 'members'>('places')

  // 인증 확인
  useEffect(() => {
    const storedCode = localStorage.getItem('adminCode')
    if (storedCode) {
      setAdminCode(storedCode)
      setIsAuthenticated(true)
    }
  }, [])

  // 인증된 경우 데이터 불러오기
  useEffect(() => {
    if (isAuthenticated && adminCode) {
      loadPendingPlaces()
      loadMembers()
    }
  }, [isAuthenticated, adminCode])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError(null)
    setLoading(true)

    try {
      // 인증 테스트 - pending 장소 목록 요청
      await adminApi.getPendingPlaces(inputCode)

      // 성공하면 저장
      localStorage.setItem('adminCode', inputCode)
      setAdminCode(inputCode)
      setIsAuthenticated(true)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthError('잘못된 관리자 코드입니다.')
      } else {
        setAuthError('인증 중 오류가 발생했습니다.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('adminCode')
    setAdminCode('')
    setIsAuthenticated(false)
    setInputCode('')
    setPendingPlaces([])
    setMembers([])
    setSelectedBrands({})
  }

  const loadMembers = async () => {
    setMembersLoading(true)
    try {
      const memberList = await adminApi.getAllMembers(adminCode)
      setMembers(memberList)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleLogout()
        setAuthError('인증이 만료되었습니다. 다시 로그인해주세요.')
      } else {
        console.error('회원 목록 조회 실패:', err)
      }
    } finally {
      setMembersLoading(false)
    }
  }

  const handleRestrict = async (memberId: string) => {
    if (!confirm('이 회원의 적립을 제한하시겠습니까?')) return

    try {
      await adminApi.restrictMember(memberId, adminCode)
      await loadMembers()
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleLogout()
        setAuthError('인증이 만료되었습니다.')
      } else {
        alert('적립 제한 처리 중 오류가 발생했습니다.')
      }
    }
  }

  const handleUnrestrict = async (memberId: string) => {
    if (!confirm('이 회원의 적립 제한을 해제하시겠습니까?')) return

    try {
      await adminApi.unrestrictMember(memberId, adminCode)
      await loadMembers()
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleLogout()
        setAuthError('인증이 만료되었습니다.')
      } else {
        alert('적립 제한 해제 처리 중 오류가 발생했습니다.')
      }
    }
  }

  const loadPendingPlaces = async () => {
    setLoading(true)
    setError(null)

    try {
      const places = await adminApi.getPendingPlaces(adminCode)
      setPendingPlaces(places)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        // 인증 실패 시 로그아웃
        handleLogout()
        setAuthError('인증이 만료되었습니다. 다시 로그인해주세요.')
      } else {
        setError('장소 목록을 불러오는데 실패했습니다.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleActivate = async (placeId: number) => {
    // reportedBrand가 있는 경우 brand 선택 확인
    const place = pendingPlaces.find(p => p.id === placeId)
    if (place?.reportedBrand && !selectedBrands[placeId]) {
      alert('브랜드를 선택해주세요.')
      return
    }

    if (!confirm('이 장소를 승인하시겠습니까?')) return

    try {
      const brand = selectedBrands[placeId]
      await adminApi.activatePlace(placeId, adminCode, brand)
      // 목록 새로고침
      await loadPendingPlaces()
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleLogout()
        setAuthError('인증이 만료되었습니다.')
      } else {
        alert('승인 처리 중 오류가 발생했습니다.')
      }
    }
  }

  const handleReject = async (placeId: number) => {
    if (!confirm('이 장소를 거부하시겠습니까?')) return

    try {
      await adminApi.rejectPlace(placeId, adminCode)
      // 목록 새로고침
      await loadPendingPlaces()
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleLogout()
        setAuthError('인증이 만료되었습니다.')
      } else {
        alert('거부 처리 중 오류가 발생했습니다.')
      }
    }
  }

  const handleBrandChange = (placeId: number, brand: string) => {
    setSelectedBrands(prev => ({
      ...prev,
      [placeId]: brand
    }))
  }

  // 로그인 페이지
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
        <div className="max-w-md w-full">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="text-center mb-8">
              <h1 className="text-2xl font-bold text-gray-900 mb-2">관리자 로그인</h1>
              <p className="text-sm text-gray-600">관리자 코드를 입력해주세요</p>
            </div>

            <form onSubmit={handleLogin} className="space-y-6">
              <div>
                <label htmlFor="adminCode" className="block text-sm font-medium text-gray-700 mb-2">
                  관리자 코드
                </label>
                <input
                  type="password"
                  id="adminCode"
                  value={inputCode}
                  onChange={(e) => setInputCode(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                  placeholder="관리자 코드를 입력하세요"
                  required
                />
              </div>

              {authError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-sm text-red-600">{authError}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? '인증 중...' : '로그인'}
              </button>
            </form>

            <div className="mt-6 text-center">
              <button
                onClick={() => router.push('/')}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                홈으로 돌아가기
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // 관리자 대시보드
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">관리자 대시보드</h1>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => {
                if (activeTab === 'places') loadPendingPlaces()
                else loadMembers()
              }}
              disabled={loading || membersLoading}
              className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
            >
              {(loading || membersLoading) ? '새로고침 중...' : '새로고침'}
            </button>
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
            >
              로그아웃
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-4 border-b border-gray-200">
            <button
              onClick={() => setActiveTab('places')}
              className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'places'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              장소 제보 검수 ({pendingPlaces.length})
            </button>
            <button
              onClick={() => setActiveTab('members')}
              className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'members'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              회원 관리 ({members.length})
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Places Tab */}
        {activeTab === 'places' && (
          <>
            {/* Stats */}
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">검수 대기 중인 장소</p>
                  <p className="text-3xl font-bold text-gray-900 mt-1">{pendingPlaces.length}개</p>
                </div>
                <div className="bg-yellow-100 p-4 rounded-full">
                  <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Places List */}
            <div className="space-y-4">
              {loading && pendingPlaces.length === 0 ? (
                <div className="bg-white rounded-lg shadow p-12 text-center">
                  <div className="animate-spin w-12 h-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
                  <p className="text-gray-600">로딩 중...</p>
                </div>
              ) : pendingPlaces.length === 0 ? (
                <div className="bg-white rounded-lg shadow p-12 text-center">
                  <div className="text-gray-400 text-5xl mb-4">✓</div>
                  <p className="text-gray-600 text-lg">검수 대기 중인 장소가 없습니다</p>
                </div>
              ) : (
                pendingPlaces.map((place) => (
                  <div key={place.id} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
                    <div className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-gray-900 mb-1">{place.name}</h3>
                          <p className="text-sm text-gray-600 mb-2">{place.address}</p>
                          {place.description && (
                            <p className="text-sm text-gray-700 mb-3">{place.description}</p>
                          )}
                        </div>
                        <span className="bg-yellow-100 text-yellow-800 text-xs font-medium px-3 py-1 rounded-full">
                          대기중
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                        <div>
                          <span className="text-gray-600">카테고리:</span>
                          <span className="ml-2 font-medium text-gray-900">
                            {place.category === 'STORE' ? '상점' : '시설'}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">타입:</span>
                          <span className="ml-2 font-medium text-gray-900">
                            {place.types.join(', ')}
                          </span>
                        </div>
                        {place.reportedBrand && (
                          <div className="col-span-2">
                            <span className="text-gray-600">제보된 서비스명:</span>
                            <span className="ml-2 font-semibold text-blue-600">{place.reportedBrand}</span>
                          </div>
                        )}
                        {place.contact && (
                          <div>
                            <span className="text-gray-600">연락처:</span>
                            <span className="ml-2 font-medium text-gray-900">{place.contact}</span>
                          </div>
                        )}
                        {place.location && (
                          <div>
                            <span className="text-gray-600">좌표:</span>
                            <span className="ml-2 font-medium text-gray-900">
                              {place.location.lat.toFixed(6)}, {place.location.lng.toFixed(6)}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* 브랜드 선택 (reportedBrand가 있을 때만 표시) */}
                      {place.reportedBrand && (
                        <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            브랜드 선택 (승인 시 필수)
                          </label>
                          <select
                            value={selectedBrands[place.id] || ''}
                            onChange={(e) => handleBrandChange(place.id, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="">선택하세요</option>
                            <option value="SUNHWA">선화</option>
                            <option value="UTURN">유턴</option>
                          </select>
                        </div>
                      )}

                      <div className="flex gap-3 pt-4 border-t border-gray-200">
                        <button
                          onClick={() => handleActivate(place.id)}
                          className="flex-1 bg-green-600 text-white py-2.5 px-4 rounded-lg font-medium hover:bg-green-700 transition-colors"
                        >
                          승인
                        </button>
                        <button
                          onClick={() => handleReject(place.id)}
                          className="flex-1 bg-red-600 text-white py-2.5 px-4 rounded-lg font-medium hover:bg-red-700 transition-colors"
                        >
                          거부
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        )}

        {/* Members Tab */}
        {activeTab === 'members' && (
          <>
            {/* Stats */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">전체 회원</p>
                    <p className="text-3xl font-bold text-gray-900 mt-1">{members.length}명</p>
                  </div>
                  <div className="bg-blue-100 p-4 rounded-full">
                    <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                    </svg>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">적립 제한 회원</p>
                    <p className="text-3xl font-bold text-red-600 mt-1">
                      {members.filter(m => m.receiptRestricted).length}명
                    </p>
                  </div>
                  <div className="bg-red-100 p-4 rounded-full">
                    <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>

            {/* Members List */}
            <div className="bg-white rounded-lg shadow overflow-hidden">
              {membersLoading && members.length === 0 ? (
                <div className="p-12 text-center">
                  <div className="animate-spin w-12 h-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
                  <p className="text-gray-600">로딩 중...</p>
                </div>
              ) : members.length === 0 ? (
                <div className="p-12 text-center">
                  <div className="text-gray-400 text-5xl mb-4">👤</div>
                  <p className="text-gray-600 text-lg">등록된 회원이 없습니다</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-200">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          회원
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          포인트
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          적립 횟수
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          최근 3일
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          마지막 적립
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          상태
                        </th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                          관리
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {members.map((member) => {
                        const isSuspicious = member.receipts3Days >= 5  // 최근 3일 적립 5회 이상이면 의심
                        const rowBgClass = member.receiptRestricted 
                          ? 'bg-red-50' 
                          : isSuspicious 
                            ? 'bg-amber-50' 
                            : ''
                        
                        return (
                          <tr key={member.id} className={rowBgClass}>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div>
                                <div className="text-sm font-medium text-gray-900">
                                  {member.nickname}
                                  {isSuspicious && !member.receiptRestricted && (
                                    <span className="ml-2 text-amber-600" title="이상 적립 의심">⚠️</span>
                                  )}
                                </div>
                                <div className="text-xs text-gray-500">{member.id.slice(0, 8)}...</div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className="text-sm font-semibold text-primary">
                                {member.pointBalance.toLocaleString()}P
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {member._count.receipts}회
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`text-sm font-semibold ${
                                member.receipts3Days >= 5 
                                  ? 'text-red-600' 
                                  : member.receipts3Days >= 3 
                                    ? 'text-amber-600' 
                                    : 'text-gray-900'
                              }`}>
                                {member.receipts3Days}회
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {member.lastReceiptAt 
                                ? new Date(member.lastReceiptAt).toLocaleString('ko-KR')
                                : '-'
                              }
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              {member.receiptRestricted ? (
                                <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full">
                                  적립 제한
                                </span>
                              ) : (
                                <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                                  정상
                                </span>
                              )}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right">
                              {member.receiptRestricted ? (
                                <button
                                  onClick={() => handleUnrestrict(member.id)}
                                  className="px-3 py-1.5 text-xs font-medium bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                                >
                                  제한 해제
                                </button>
                              ) : (
                                <button
                                  onClick={() => handleRestrict(member.id)}
                                  className="px-3 py-1.5 text-xs font-medium bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                                >
                                  적립 제한
                                </button>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
