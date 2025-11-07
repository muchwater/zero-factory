'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import KakaoMap from '@/components/KakaoMap'
import SearchBar from '@/components/SearchBar'
import CategoryCard from '@/components/CategoryCard'
import PlaceCard from '@/components/PlaceCard'
import BottomNavigation from '@/components/BottomNavigation'
import MapOverlay from '@/components/MapOverlay'
import { usePlaces } from '@/hooks/usePlaces'
import type { Place, PlaceNearby } from '@/types/api'

export default function Home() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'camera' | 'profile'>('home')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null)
  const [selectedPlace, setSelectedPlace] = useState<Place | PlaceNearby | null>(null)

  // 기본 위치 (대전 유성구)
  const defaultLocation = { lat: 36.3731, lng: 127.362 }
  const currentLocation = userLocation || defaultLocation

  // 장소 데이터 가져오기 - 전체 장소 조회
  const { 
    places, 
    nearbyPlaces, 
    loading, 
    error, 
    fetchAllPlaces,
    fetchNearbyPlaces 
  } = usePlaces({
    autoFetch: true  // 자동으로 전체 장소 조회
  })

  // 사용자 위치 가져오기
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          })
        },
        (error) => {
          console.warn('위치 정보를 가져올 수 없습니다:', error)
        }
      )
    }
  }, [])

  const categories = [
    {
      id: 'coffee',
      title: '다회용컵 카페',
      description: '재사용 컵을 사용하는 친환경 카페',
      icon: 'coffee' as const,
    },
    {
      id: 'recycle',
      title: '다회용컵 반납함',
      description: '사용한 컵을 반납할 수 있는 곳',
      icon: 'recycle' as const,
    },
    {
      id: 'store',
      title: '텀블러 포인트 적립',
      description: '텀블러 사용 시 포인트를 적립받는 매장',
      icon: 'store' as const,
    },
    {
      id: 'wash',
      title: '텀블러 세척기',
      description: '텀블러를 깨끗하게 세척할 수 있는 곳',
      icon: 'wash' as const,
    },
  ]

  // 카테고리 ID와 PlaceType 매핑
  const categoryToType: Record<string, 'RENT' | 'RETURN' | 'BONUS' | 'CLEAN'> = {
    'coffee': 'RENT',
    'recycle': 'RETURN',
    'store': 'BONUS',
    'wash': 'CLEAN',
  }

  // 선택된 카테고리에 따라 장소 필터링
  const filteredPlaces = selectedCategory
    ? places.filter(place => place.types.includes(categoryToType[selectedCategory] as any))
    : places

  // API에서 가져온 장소 데이터를 PlaceCard 형식으로 변환
  const displayPlaces = filteredPlaces.map((place) => ({
    id: place.id,
    name: place.name,
    distance: '-', // 전체 조회이므로 거리 정보 없음
    status: 'open' as const, // 실제로는 운영시간을 확인해야 함
    rating: 4.5, // 실제로는 리뷰 데이터가 있어야 함
    category: place.types.join(', '),
    icon: place.types.includes('RENT') ? '☕' : 
          place.types.includes('RETURN') ? '♻️' :
          place.types.includes('BONUS') ? '🏪' : '🧼',
  }))

  const handleSearch = (value: string) => {
    console.log('검색:', value)
  }

  const handleCategoryClick = (categoryId: string) => {
    setSelectedCategory(selectedCategory === categoryId ? null : categoryId)
  }

  const handlePlaceClick = (placeId: number) => {
    const place = filteredPlaces.find(p => p.id === placeId)
    if (place) {
      setSelectedPlace(place)
      console.log('장소 클릭:', place)
    }
  }

  const handleMapPlaceClick = (place: Place | PlaceNearby) => {
    setSelectedPlace(place)
    console.log('맵에서 장소 클릭:', place)
  }

  const handleFilterClick = () => {
    console.log('필터 클릭')
  }

  const handleAddPlaceClick = () => {
    router.push('/add-place')
  }

  const handleTabChange = (tab: 'home' | 'search' | 'camera' | 'profile') => {
    setActiveTab(tab)

    // 탭에 따라 페이지 이동
    switch (tab) {
      case 'camera':
        router.push('/camera')
        break
      case 'home':
      case 'search':
      case 'profile':
        // 현재 페이지에 머무름 (향후 각 페이지가 생기면 라우팅 추가)
        break
    }
  }

  return (
    <div className="bg-background min-h-screen flex flex-col pb-20">
      {/* Header */}
      <div className="bg-white border-b border-border sticky top-0 z-40">
        <div className="px-4 py-3">
          <h1 className="text-xl font-bold text-black text-center">제로팩토리</h1>
          <p className="text-sm text-muted text-center mt-1">친환경 라이프스타일 가이드</p>
        </div>
      </div>

      {/* Search Section */}
      <div className="px-4 py-4 bg-white">
        <SearchBar 
          placeholder="상점명 또는 지역명으로 검색..."
          onSearch={handleSearch}
        />
      </div>

          {/* Map Section */}
          <div className="px-4 py-2">
            <div className="relative h-80 rounded-2xl overflow-hidden shadow-lg">
              <KakaoMap
                width="100%"
                height="320px"
                className="w-full h-full"
                center={currentLocation}
                level={3}
                places={filteredPlaces}
                onPlaceClick={handleMapPlaceClick}
              />
              <MapOverlay
                onFilterClick={handleFilterClick}
                onAddPlaceClick={handleAddPlaceClick}
              />
              
              {/* 로딩 상태 표시 */}
              {loading && (
                <div className="absolute inset-0 bg-white/80 flex items-center justify-center rounded-2xl">
                  <div className="text-center">
                    <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
                    <div className="text-sm text-muted">장소 정보를 불러오는 중...</div>
                  </div>
                </div>
              )}
              
              {/* 에러 상태 표시 */}
              {error && (
                <div className="absolute inset-0 bg-red-50/90 flex items-center justify-center rounded-2xl">
                  <div className="text-center p-4">
                    <div className="text-red-500 text-lg mb-2">⚠️</div>
                    <div className="text-sm text-red-600 mb-2">데이터 로드 실패</div>
                    <div className="text-xs text-red-500">{error}</div>
                    <button 
                      onClick={() => fetchAllPlaces()}
                      className="mt-2 px-3 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600"
                    >
                      다시 시도
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

      {/* Category Section */}
      <div className="px-4 py-4">
        <h2 className="text-lg font-semibold text-foreground mb-3">카테고리</h2>
        <div className="grid grid-cols-2 gap-3">
          {categories.map((category) => (
            <CategoryCard
              key={category.id}
              title={category.title}
              icon={category.icon}
              description={category.description}
              isActive={selectedCategory === category.id}
              onClick={() => handleCategoryClick(category.id)}
            />
          ))}
        </div>
      </div>

      {/* Nearby Places Section */}
      <div className="px-4 py-4 flex-1">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-foreground">제로웨이스트 장소</h2>
          <button className="text-sm text-primary font-medium hover:text-primary-dark transition-colors">
            전체보기
          </button>
        </div>
        
            <div className="space-y-3">
              {displayPlaces.length > 0 ? (
                displayPlaces.map((place) => (
                  <PlaceCard
                    key={place.id}
                    name={place.name}
                    distance={place.distance}
                    status={place.status}
                    rating={place.rating}
                    category={place.category}
                    icon={place.icon}
                    onClick={() => handlePlaceClick(place.id)}
                  />
                ))
              ) : (
                <div className="text-center py-8">
                  <div className="text-muted text-sm">
                    {loading ? '장소 정보를 불러오는 중...' : '등록된 장소가 없습니다.'}
                  </div>
                </div>
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
