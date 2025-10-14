'use client'

import { useState } from 'react'
import KakaoMap from '@/components/KakaoMap'
import SearchBar from '@/components/SearchBar'
import CategoryCard from '@/components/CategoryCard'
import PlaceCard from '@/components/PlaceCard'
import BottomNavigation from '@/components/BottomNavigation'
import MapOverlay from '@/components/MapOverlay'

export default function Home() {
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'profile'>('home')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

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

  const nearbyPlaces = [
    {
      id: 1,
      name: '에코프렌들리 카페',
      distance: '0.5km',
      status: 'open' as const,
      rating: 4.8,
      category: '다회용컵 카페',
      icon: '♻️',
    },
    {
      id: 2,
      name: '그린 리필 스테이션',
      distance: '1.2km',
      status: 'open' as const,
      rating: 4.6,
      category: '텀블러 포인트 적립',
      icon: '🏪',
    },
    {
      id: 3,
      name: '제로웨이스트 센터',
      distance: '2.1km',
      status: 'closed' as const,
      rating: 4.9,
      category: '다회용컵 반납함',
      icon: '🗑️',
    },
    {
      id: 4,
      name: '클린 워시 스테이션',
      distance: '0.8km',
      status: 'open' as const,
      rating: 4.7,
      category: '텀블러 세척기',
      icon: '🧼',
    },
  ]

  const handleSearch = (value: string) => {
    console.log('검색:', value)
  }

  const handleCategoryClick = (categoryId: string) => {
    setSelectedCategory(selectedCategory === categoryId ? null : categoryId)
  }

  const handlePlaceClick = (placeId: number) => {
    console.log('장소 클릭:', placeId)
  }

  const handleFilterClick = () => {
    console.log('필터 클릭')
  }

  const handleAddPlaceClick = () => {
    console.log('장소 추가 클릭')
  }

  return (
    <div className="bg-background min-h-screen flex flex-col pb-20">
      {/* Header */}
      <div className="bg-white border-b border-border sticky top-0 z-40">
        <div className="px-4 py-3">
          <h1 className="text-xl font-bold text-foreground text-center">제로팩토리</h1>
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
          />
          <MapOverlay 
            onFilterClick={handleFilterClick}
            onAddPlaceClick={handleAddPlaceClick}
          />
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
          <h2 className="text-lg font-semibold text-foreground">근처 제로웨이스트</h2>
          <button className="text-sm text-primary font-medium hover:text-primary-dark transition-colors">
            전체보기
          </button>
        </div>
        
        <div className="space-y-3">
          {nearbyPlaces.map((place) => (
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
          ))}
        </div>
      </div>

      {/* Bottom Navigation */}
      <BottomNavigation 
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
    </div>
  )
}
