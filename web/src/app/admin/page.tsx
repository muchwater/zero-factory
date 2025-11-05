'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { adminApi, ApiError } from '@/services/api'
import type { Place } from '@/types/api'

export default function AdminPage() {
  const router = useRouter()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [adminCode, setAdminCode] = useState('')
  const [inputCode, setInputCode] = useState('')
  const [pendingPlaces, setPendingPlaces] = useState<Place[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [authError, setAuthError] = useState<string | null>(null)

  // 인증 확인
  useEffect(() => {
    const storedCode = localStorage.getItem('adminCode')
    if (storedCode) {
      setAdminCode(storedCode)
      setIsAuthenticated(true)
    }
  }, [])

  // 인증된 경우 pending 장소 목록 불러오기
  useEffect(() => {
    if (isAuthenticated && adminCode) {
      loadPendingPlaces()
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
    if (!confirm('이 장소를 승인하시겠습니까?')) return

    try {
      await adminApi.activatePlace(placeId, adminCode)
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
            <p className="text-sm text-gray-600 mt-1">장소 제보 검수</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={loadPendingPlaces}
              disabled={loading}
              className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
            >
              {loading ? '새로고침 중...' : '새로고침'}
            </button>
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
            >
              로그아웃
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
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

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

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
      </div>
    </div>
  )
}
