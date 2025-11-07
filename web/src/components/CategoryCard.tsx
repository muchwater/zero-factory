'use client'

import { CoffeeIcon, RecycleIcon, StoreIcon, WashIcon } from './IconComponents'

interface CategoryCardProps {
  title: string
  icon: 'coffee' | 'recycle' | 'store' | 'wash'
  description?: string
  onClick?: () => void
  isActive?: boolean
}

const iconMap = {
  coffee: CoffeeIcon,
  recycle: RecycleIcon,
  store: StoreIcon,
  wash: WashIcon,
}

export default function CategoryCard({ 
  title, 
  icon, 
  description, 
  onClick, 
  isActive = false 
}: CategoryCardProps) {
  const IconComponent = iconMap[icon]
  
  return (
    <button
      onClick={onClick}
      className={`
        group relative overflow-hidden rounded-xl border-2 transition-all duration-300 ease-in-out
        ${isActive 
          ? 'border-primary bg-primary/5 shadow-lg scale-105' 
          : 'border-border bg-white hover:border-primary/50 hover:shadow-md hover:scale-102'
        }
        p-4 w-full text-left
      `}
    >
      <div className="flex items-center space-x-3">
        <div className={`
          flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center transition-colors duration-300
          ${isActive 
            ? 'bg-primary text-white' 
            : 'bg-secondary text-muted group-hover:bg-primary/10 group-hover:text-primary'
          }
        `}>
          <IconComponent className="w-6 h-6" />
        </div>
        
        <div className="flex-1 min-w-0">
          <h3 className={`
            font-semibold text-sm transition-colors duration-300
            ${isActive ? 'text-primary' : 'text-black group-hover:text-primary'}
          `}>
            {title}
          </h3>
          {description && (
            <p className="text-xs text-muted mt-1 line-clamp-2">
              {description}
            </p>
          )}
        </div>
      </div>
      
      {/* 활성 상태 표시 */}
      {isActive && (
        <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full animate-bounce-gentle" />
      )}
    </button>
  )
}
