'use client'

import { useState, useEffect, Suspense, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import BottomNavigation from '@/components/BottomNavigation'
import { useMember } from '@/hooks/useMember'
import { receiptsApi } from '@/services/api'
import { compressImage } from '@/utils/imageCompression'
import { formatDistance, calculateDistance, type Coordinates } from '@/utils/location'
import type { PlaceNearby } from '@/types/api'

// 검색된 장소 타입
interface SearchedPlace {
  name: string
  address: string
  lat: number
  lng: number
  distance: number // 현재 위치로부터의 거리 (미터)
}

function ZeroReceiptContent() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'profile'>('search')
  const [productDescription, setProductDescription] = useState('')
  const [photo, setPhoto] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [nearbyPlaces, setNearbyPlaces] = useState<PlaceNearby[]>([])
  const [selectedPlaceId, setSelectedPlaceId] = useState<number | null>(null)
  const { member, loading, refreshMember } = useMember()

  // 새로운 상태들
  const [currentPosition, setCurrentPosition] = useState<Coordinates | null>(null)
  const [isKakaoLoaded, setIsKakaoLoaded] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchedPlace[]>([])
  const [showSearchResults, setShowSearchResults] = useState(false)
  const [selectedSearchedPlace, setSelectedSearchedPlace] = useState<SearchedPlace | null>(null)
  const [searchError, setSearchError] = useState<string | null>(null)
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Kakao Maps SDK 로드 확인
  useEffect(() => {
    const checkKakaoLoaded = () => {
      if (window.kakao && window.kakao.maps) {
        window.kakao.maps.load(() => {
          setIsKakaoLoaded(true)
        })
      } else {
        setTimeout(checkKakaoLoaded, 100)
      }
    }
    checkKakaoLoaded()
  }, [])

  // 카메라 페이지에서 sessionStorage로 전달된 데이터 복원
  useEffect(() => {
    if (typeof window === 'undefined') return

    const photoData = sessionStorage.getItem('zeroReceiptPhoto')
    const placesData = sessionStorage.getItem('zeroReceiptNearbyPlaces')
    const positionData = sessionStorage.getItem('zeroReceiptCurrentPosition')

    if (photoData) {
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
          sessionStorage.removeItem('zeroReceiptPhoto')
        })
    }

    if (placesData) {
      try {
        const places: PlaceNearby[] = JSON.parse(placesData)
        setNearbyPlaces(places)
        // 가장 가까운 장소를 기본 선택
        if (places.length > 0) {
          setSelectedPlaceId(places[0].id)
        }
      } catch (err) {
        console.error('failed to parse nearby places:', err)
      } finally {
        sessionStorage.removeItem('zeroReceiptNearbyPlaces')
      }
    }

    // 현재 위치 복원
    if (positionData) {
      try {
        const position: Coordinates = JSON.parse(positionData)
        setCurrentPosition(position)
      } catch (err) {
        console.error('failed to parse current position:', err)
      } finally {
        sessionStorage.removeItem('zeroReceiptCurrentPosition')
      }
    }

    // 검증 결과 정리
    sessionStorage.removeItem('zeroReceiptVerificationResult')
  }, [])

  const handleTabChange = (tab: 'home' | 'search' | 'profile') => {
    setActiveTab(tab)
    if (tab === 'home') {
      router.push('/')
    } else if (tab === 'profile') {
      router.push('/profile')
    }
  }

  // 카카오 장소 검색
  const searchPlaces = useCallback(async (query: string) => {
    if (!query.trim() || !isKakaoLoaded || !currentPosition) {
      setSearchResults([])
      setShowSearchResults(false)
      return
    }

    try {
      const places = new window.kakao.maps.services.Places()
      
      places.keywordSearch(query, (results: any, status: any) => {
        if (status === window.kakao.maps.services.Status.OK) {
          const searchedPlaces: SearchedPlace[] = results.map((item: any) => {
            const lat = parseFloat(item.y)
            const lng = parseFloat(item.x)
            const distance = calculateDistance(currentPosition, { lat, lng })
            return {
              name: item.place_name,
              address: item.address_name,
              lat,
              lng,
              distance,
            }
          })
          // 거리순 정렬
          searchedPlaces.sort((a, b) => a.distance - b.distance)
          setSearchResults(searchedPlaces)
          setShowSearchResults(true)
          setSearchError(null)
        } else if (status === window.kakao.maps.services.Status.ZERO_RESULT) {
          setSearchResults([])
          setShowSearchResults(true)
          setSearchError('검색 결과가 없습니다.')
        } else {
          setSearchError('검색 중 오류가 발생했습니다.')
        }
      })
    } catch (err) {
      console.error('장소 검색 오류:', err)
      setSearchError('장소 검색 중 오류가 발생했습니다.')
    }
  }, [isKakaoLoaded, currentPosition])

  // 검색어 입력 핸들러 (디바운싱 적용)
  const handleSearchInput = (value: string) => {
    setSearchQuery(value)
    
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }

    if (!value.trim()) {
      setSearchResults([])
      setShowSearchResults(false)
      return
    }

    searchTimeoutRef.current = setTimeout(() => {
      searchPlaces(value)
    }, 300)
  }

  // 검색된 장소 선택
  const selectSearchedPlace = (place: SearchedPlace) => {
    if (place.distance > 100) {
      alert(`이 장소는 현재 위치에서 ${formatDistance(place.distance)} 떨어져 있습니다.\n100m 이내의 장소만 선택할 수 있습니다.`)
      return
    }
    setSelectedSearchedPlace(place)
    setSelectedPlaceId(null) // 등록된 장소 선택 해제
    setShowSearchResults(false)
    setSearchQuery(place.name)
  }

  // 등록된 장소 선택
  const selectRegisteredPlace = (placeId: number) => {
    setSelectedPlaceId(placeId)
    setSelectedSearchedPlace(null) // 검색된 장소 선택 해제
    setSearchQuery('')
  }

  // 적립 가능 여부 체크 (쿨다운 10분)
  const COOLDOWN_MINUTES = 10

  const checkReceiptEligibility = (): { canSubmit: boolean; message?: string; remainingMinutes?: number } => {
    if (!member) {
      return { canSubmit: false, message: '회원 정보를 불러오는 중입니다.' }
    }

    // 적립 제한 여부 확인
    if (member.receiptRestricted) {
      return { canSubmit: false, message: '적립이 제한된 회원입니다. 관리자에게 문의하세요.' }
    }

    // 쿨다운 체크
    if (member.lastReceiptAt) {
      const lastReceiptTime = new Date(member.lastReceiptAt).getTime()
      const cooldownEndTime = lastReceiptTime + COOLDOWN_MINUTES * 60 * 1000
      const now = Date.now()

      if (now < cooldownEndTime) {
        const remainingMs = cooldownEndTime - now
        const remainingMinutes = Math.ceil(remainingMs / 60000)
        return { 
          canSubmit: false, 
          message: `${remainingMinutes}분 후에 다시 적립할 수 있습니다.`,
          remainingMinutes 
        }
      }
    }

    return { canSubmit: true }
  }

  const handleCameraClick = () => {
    const eligibility = checkReceiptEligibility()
    
    if (!eligibility.canSubmit) {
      alert(eligibility.message)
      return
    }

    router.push('/zero-receipt/camera')
  }

  const handleSubmit = async () => {
    // 등록된 장소 또는 검색된 장소 중 하나는 선택되어야 함
    if (!member || !photo || (!selectedPlaceId && !selectedSearchedPlace)) {
      return
    }

    setSubmitting(true)
    try {
      // 이미지 압축 (1200px, 70% 품질)
      const compressedPhoto = await compressImage(photo, 1200, 0.7)

      // 검색된 장소인 경우 placeId 없이, 등록된 장소는 placeId와 함께 제출
      await receiptsApi.submitReceipt({
        memberId: member.id,
        productDescription: selectedSearchedPlace 
          ? `${productDescription} [${selectedSearchedPlace.name}]`.trim()
          : productDescription,
        photoFile: compressedPhoto,
        placeId: selectedPlaceId || undefined,
      })

      // 포인트 잔액 새로고침
      await refreshMember()

      // 인증 완료 상태로 변경
      setSubmitted(true)
    } catch (error: any) {
      console.error('영수증 제출 실패:', error)
      
      // 서버 에러 메시지 파싱
      if (error.status === 400) {
        // 쿨다운 에러 (10분 제한)
        alert('아직 적립할 수 없습니다. 이전 적립 후 10분이 지나야 다시 적립할 수 있습니다.')
      } else if (error.status === 403) {
        // 적립 제한 회원
        alert('적립이 제한된 회원입니다. 관리자에게 문의하세요.')
      } else {
        alert('영수증 제출에 실패했습니다. 다시 시도해주세요.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  const handleReset = () => {
    setSubmitted(false)
    setProductDescription('')
    setPhoto(null)
    setNearbyPlaces([])
    setSelectedPlaceId(null)
    setCurrentPosition(null)
    setSearchQuery('')
    setSearchResults([])
    setSelectedSearchedPlace(null)
    setShowSearchResults(false)
  }

  // 인증 완료 화면
  if (submitted) {
    return (
      <div className="bg-background min-h-screen flex flex-col pb-20">
        {/* Header */}
        <div className="bg-white border-b border-border sticky top-0 z-40">
          <div className="px-4 py-3">
            <h1 className="text-xl font-bold text-black text-center">제로영수증</h1>
          </div>
        </div>

        {/* Success Content */}
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-8">
          <div className="text-center">
            <div className="text-8xl mb-6">✅</div>
            <h2 className="text-2xl font-bold text-green-600 mb-3">인증 완료!</h2>
            <p className="text-gray-600 mb-2">
              다회용기 사용이 인증되었습니다.
            </p>
            <p className="text-lg font-semibold text-primary mb-8">
              +100 포인트가 적립되었습니다!
            </p>
            
            <div className="space-y-3">
              <button
                onClick={() => router.push('/profile')}
                className="w-full px-8 py-3 bg-primary text-white rounded-xl font-bold hover:bg-primary-dark transition-colors"
              >
                내 프로필 보기
              </button>
              <button
                onClick={() => router.push('/')}
                className="w-full px-8 py-3 bg-gray-200 text-gray-700 rounded-xl font-medium hover:bg-gray-300 transition-colors"
              >
                홈으로 돌아가기
              </button>
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

  // 인증 데이터가 없는 경우 (카메라 촬영 필요)
  // photo와 currentPosition이 있으면 진행 가능 (등록된 장소가 없어도 검색으로 선택 가능)
  const hasVerificationData = photo && currentPosition

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
        {!hasVerificationData ? (
          // 카메라 촬영 유도 화면
          (() => {
            const eligibility = checkReceiptEligibility()
            const isRestricted = member?.receiptRestricted
            const hasCooldown = !eligibility.canSubmit && eligibility.remainingMinutes

            return (
              <div className="flex flex-col items-center justify-center py-12">
                {isRestricted ? (
                  // 적립 제한 회원
                  <>
                    <div className="text-6xl mb-6">🚫</div>
                    <h2 className="text-xl font-bold text-red-600 mb-3 text-center">
                      적립이 제한되었습니다
                    </h2>
                    <p className="text-gray-500 text-sm text-center mb-8 max-w-xs">
                      관리자에게 문의하세요.
                    </p>
                  </>
                ) : hasCooldown ? (
                  // 쿨다운 중
                  <>
                    <div className="text-6xl mb-6">⏳</div>
                    <h2 className="text-xl font-bold text-orange-600 mb-3 text-center">
                      잠시 후 다시 시도해주세요
                    </h2>
                    <p className="text-gray-500 text-sm text-center mb-4 max-w-xs">
                      이전 적립 후 10분이 지나야<br />
                      다시 적립할 수 있습니다.
                    </p>
                    <div className="bg-orange-100 text-orange-700 px-6 py-3 rounded-xl font-bold text-lg mb-8">
                      {eligibility.remainingMinutes}분 후 적립 가능
                    </div>
                    <button
                      onClick={() => refreshMember()}
                      className="text-sm text-gray-500 underline hover:text-gray-700"
                    >
                      새로고침
                    </button>
                  </>
                ) : (
                  // 적립 가능
                  <>
                    <div className="text-6xl mb-6">📸</div>
                    <h2 className="text-xl font-bold text-foreground mb-3 text-center">
                      다회용기를 촬영해주세요
                    </h2>
                    <p className="text-gray-500 text-sm text-center mb-8 max-w-xs">
                      다회용기에 음료가 담긴 사진을 촬영하면<br />
                      AI가 자동으로 검증하고 포인트를 적립해드립니다.
                    </p>
                    <button
                      onClick={handleCameraClick}
                      disabled={loading}
                      className="flex items-center gap-3 px-8 py-4 bg-primary text-white rounded-2xl font-bold text-lg hover:bg-primary-dark transition-colors shadow-lg disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                      <svg
                        className="w-6 h-6"
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
                      {loading ? '로딩 중...' : '촬영하기'}
                    </button>
                  </>
                )}
              </div>
            )
          })()
        ) : (
          // 적립 폼 화면
          <>
            {/* Section Title */}
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-foreground">
                인증이 완료되었습니다!
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                아래 정보를 확인하고 적립을 완료해주세요.
              </p>
            </div>

            {/* Step 1: Product Description */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-foreground mb-2">
                구입한 제품/서비스 (선택사항)
              </label>
              <input
                type="text"
                value={productDescription}
                onChange={(e) => setProductDescription(e.target.value)}
                placeholder="예: 아메리카노"
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
              />
            </div>

            {/* Step 2: Place Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-foreground mb-2">
                사용 장소 선택
              </label>

              {/* 등록된 근처 장소가 있는 경우 */}
              {nearbyPlaces.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs text-gray-500 mb-2">등록된 장소 (100m 이내)</p>
                  <div className="space-y-2">
                    {nearbyPlaces.map((place, index) => (
                      <label
                        key={place.id}
                        className={`
                          flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-all
                          ${selectedPlaceId === place.id
                            ? 'border-primary bg-primary/5'
                            : 'border-gray-200 bg-white hover:border-gray-300'
                          }
                        `}
                      >
                        <input
                          type="radio"
                          name="place"
                          value={place.id}
                          checked={selectedPlaceId === place.id}
                          onChange={() => selectRegisteredPlace(place.id)}
                          className="w-4 h-4 text-primary focus:ring-primary"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground truncate">
                              {place.name}
                            </span>
                            {index === 0 && (
                              <span className="px-2 py-0.5 text-xs bg-primary/10 text-primary rounded-full">
                                가장 가까움
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-gray-500 mt-0.5 flex items-center gap-2">
                            <span>{formatDistance(place.distance)}</span>
                            <span>•</span>
                            <span className="truncate">{place.address}</span>
                          </div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* 장소 검색 섹션 */}
              <div className="relative">
                <p className="text-xs text-gray-500 mb-2">
                  {nearbyPlaces.length > 0 ? '또는 장소 검색' : '장소 검색 (100m 이내)'}
                </p>
                <div className="relative">
                  <div className="flex items-center gap-2 px-3 py-2.5 border border-gray-300 rounded-lg bg-white">
                    <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => handleSearchInput(e.target.value)}
                      onFocus={() => searchResults.length > 0 && setShowSearchResults(true)}
                      placeholder="매장명, 주소로 검색..."
                      className="flex-1 text-sm placeholder-gray-400 focus:outline-none"
                    />
                    {searchQuery && (
                      <button
                        type="button"
                        onClick={() => {
                          setSearchQuery('')
                          setSearchResults([])
                          setShowSearchResults(false)
                          setSelectedSearchedPlace(null)
                        }}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>

                  {/* 검색 결과 드롭다운 */}
                  {showSearchResults && (
                    <div className="absolute top-full left-0 right-0 z-50 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto mt-1">
                      {searchError ? (
                        <div className="px-4 py-3 text-sm text-gray-500 text-center">
                          {searchError}
                        </div>
                      ) : searchResults.length === 0 ? (
                        <div className="px-4 py-3 text-sm text-gray-500 text-center">
                          검색 결과가 없습니다
                        </div>
                      ) : (
                        searchResults.map((place, index) => {
                          const isWithinRange = place.distance <= 100
                          return (
                            <button
                              key={index}
                              type="button"
                              onClick={() => selectSearchedPlace(place)}
                              className={`w-full px-4 py-3 text-left border-b border-gray-100 last:border-b-0 transition-colors ${
                                isWithinRange 
                                  ? 'hover:bg-gray-50' 
                                  : 'bg-gray-50 opacity-60'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className={`font-medium text-sm ${isWithinRange ? 'text-foreground' : 'text-gray-400'}`}>
                                  {place.name}
                                </span>
                                <span className={`text-xs px-2 py-0.5 rounded-full ${
                                  isWithinRange
                                    ? 'bg-green-100 text-green-700'
                                    : 'bg-red-100 text-red-600'
                                }`}>
                                  {formatDistance(place.distance)}
                                </span>
                              </div>
                              <div className="text-xs text-gray-500 mt-0.5 truncate">
                                {place.address}
                              </div>
                              {!isWithinRange && (
                                <div className="text-xs text-red-500 mt-1">
                                  100m 이내의 장소만 선택 가능
                                </div>
                              )}
                            </button>
                          )
                        })
                      )}
                    </div>
                  )}
                </div>

                {/* 선택된 검색 장소 표시 */}
                {selectedSearchedPlace && (
                  <div className="mt-3 p-3 bg-primary/5 border-2 border-primary rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-foreground truncate">
                            {selectedSearchedPlace.name}
                          </span>
                          <span className="px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded-full">
                            {formatDistance(selectedSearchedPlace.distance)}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mt-0.5 truncate">
                          {selectedSearchedPlace.address}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedSearchedPlace(null)
                          setSearchQuery('')
                        }}
                        className="ml-2 text-gray-400 hover:text-gray-600"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* 장소 미선택 안내 */}
              {!selectedPlaceId && !selectedSearchedPlace && nearbyPlaces.length === 0 && (
                <p className="text-xs text-orange-600 mt-2">
                  * 100m 이내에 등록된 장소가 없습니다. 위에서 장소를 검색해주세요.
                </p>
              )}
            </div>

            {/* Submit Button */}
            <div className="flex justify-center mt-8">
              {loading ? (
                <div className="text-sm text-muted">회원 정보를 불러오는 중...</div>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={(!selectedPlaceId && !selectedSearchedPlace) || !member || submitting}
                  className={`
                    w-full px-12 py-3 rounded-xl font-bold text-sm text-white
                    transition-all duration-300
                    ${(selectedPlaceId || selectedSearchedPlace) && member && !submitting
                      ? 'bg-primary hover:bg-primary-dark shadow-md hover:shadow-lg'
                      : 'bg-gray-400 cursor-not-allowed'
                    }
                  `}
                >
                  {submitting ? '적립 중...' : '포인트 적립하기'}
                </button>
              )}
            </div>

            {/* 다시 촬영 버튼 */}
            <div className="flex justify-center mt-4">
              <button
                onClick={handleReset}
                className="text-sm text-gray-500 underline hover:text-gray-700"
              >
                다시 촬영하기
              </button>
            </div>
          </>
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
