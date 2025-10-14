'use client'

import { HomeIcon, ListIcon, UserIcon } from './IconComponents'

interface BottomNavigationProps {
  activeTab: 'home' | 'search' | 'profile'
  onTabChange: (tab: 'home' | 'search' | 'profile') => void
}

export default function BottomNavigation({ activeTab, onTabChange }: BottomNavigationProps) {
  const tabs = [
    {
      id: 'search' as const,
      label: '검색',
      icon: ListIcon,
    },
    {
      id: 'home' as const,
      label: '홈',
      icon: HomeIcon,
    },
    {
      id: 'profile' as const,
      label: '프로필',
      icon: UserIcon,
    },
  ]

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-border z-50">
      <div className="flex">
        {tabs.map((tab) => {
          const IconComponent = tab.icon
          const isActive = activeTab === tab.id
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`
                flex-1 flex flex-col items-center justify-center py-2 px-1
                transition-all duration-300 ease-in-out
                ${isActive 
                  ? 'text-primary' 
                  : 'text-muted hover:text-foreground'
                }
              `}
            >
              <div className={`
                w-8 h-8 flex items-center justify-center rounded-full transition-all duration-300
                ${isActive 
                  ? 'bg-primary/10 scale-110' 
                  : 'hover:bg-secondary'
                }
              `}>
                <IconComponent className={`w-5 h-5 ${isActive ? 'scale-110' : ''}`} />
              </div>
              
              <span className={`
                text-xs font-medium mt-1 transition-all duration-300
                ${isActive ? 'scale-105' : ''}
              `}>
                {tab.label}
              </span>
              
              {/* 활성 탭 표시 */}
              {isActive && (
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-8 h-1 bg-primary rounded-b-full" />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
