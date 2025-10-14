'use client'

import { LocationIcon, ClockIcon, StarIcon, ChevronRightIcon } from './IconComponents'

interface PlaceCardProps {
  name: string
  distance: string
  status: 'open' | 'closed' | 'unknown'
  rating?: number
  category: string
  icon: string
  onClick?: () => void
}

export default function PlaceCard({ 
  name, 
  distance, 
  status, 
  rating, 
  category,
  icon,
  onClick 
}: PlaceCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open':
        return 'text-green-600 bg-green-50'
      case 'closed':
        return 'text-red-600 bg-red-50'
      default:
        return 'text-muted bg-secondary'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'open':
        return '영업중'
      case 'closed':
        return '영업종료'
      default:
        return '정보없음'
    }
  }

  return (
    <button
      onClick={onClick}
      className="group w-full p-4 rounded-xl border border-border bg-white hover:border-primary/50 hover:shadow-md transition-all duration-300 ease-in-out hover:scale-[1.02]"
    >
      <div className="flex items-center space-x-3">
        {/* 아이콘 */}
        <div className="flex-shrink-0 w-12 h-12 rounded-full bg-secondary flex items-center justify-center text-xl">
          {icon}
        </div>
        
        {/* 정보 */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-sm text-foreground group-hover:text-primary transition-colors duration-300 truncate">
                {name}
              </h3>
              <p className="text-xs text-muted mt-1 truncate">
                {category}
              </p>
              
              <div className="flex items-center space-x-2 mt-2">
                <div className="flex items-center space-x-1">
                  <LocationIcon className="w-3 h-3 text-muted" />
                  <span className="text-xs text-muted">{distance}</span>
                </div>
                
                {rating && (
                  <div className="flex items-center space-x-1">
                    <StarIcon className="w-3 h-3 text-accent fill-current" />
                    <span className="text-xs text-muted">{rating.toFixed(1)}</span>
                  </div>
                )}
              </div>
            </div>
            
            {/* 상태 및 화살표 */}
            <div className="flex items-center space-x-2">
              <span className={`
                px-2 py-1 rounded-full text-xs font-medium
                ${getStatusColor(status)}
              `}>
                {getStatusText(status)}
              </span>
              
              <ChevronRightIcon className="w-4 h-4 text-muted group-hover:text-primary transition-colors duration-300" />
            </div>
          </div>
        </div>
      </div>
    </button>
  )
}
