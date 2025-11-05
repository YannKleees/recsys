import React, { useState } from 'react';

// Mock hotel recommendation data
const mockHotel = {
  name: "Hotel Baltic",
  rating: 4.5,
  badges: ["Great Service", "Clean Rooms", "Close to Beach"],
  reasons: {
    group: "Your group values quiet locations and great service.",
    individuals: [
      { user: "Alice", reason: "Loves beachfront — 200m to the sea" },
      { user: "Bob", reason: "Gym & wellness center available" },
      { user: "Eve", reason: "High value for money — strong 'value' sub-ratings" }
    ]
  }
};

export default function HotelRecsUI() {
  const [showGroupDetails, setShowGroupDetails] = useState(true);

  return (
    <div className="p-4 max-w-xl mx-auto space-y-4">
      <div className="rounded-2xl shadow-lg p-4 space-y-2 border">
        <h2 className="text-xl font-semibold">{mockHotel.name}</h2>
        <p className="text-sm">⭐ {mockHotel.rating} average rating</p>

        <div className="flex flex-wrap gap-2 mt-2">
          {mockHotel.badges.map((badge, i) => (
            <span key={i} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs">
              {badge}
            </span>
          ))}
        </div>

        <div className="flex gap-2 mt-4">
          <button
            className={`px-3 py-1 rounded-full text-xs ${showGroupDetails ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            onClick={() => setShowGroupDetails(true)}
          >
            Group View
          </button>
          <button
            className={`px-3 py-1 rounded-full text-xs ${!showGroupDetails ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            onClick={() => setShowGroupDetails(false)}
          >
            Individual View
          </button>
        </div>

        {showGroupDetails ? (
          <div className="mt-3 text-sm">
            <p className="font-medium">Group Explanation</p>
            <p>{mockHotel.reasons.group}</p>
          </div>
        ) : (
          <div className="mt-3 text-sm space-y-1">
            <p className="font-medium">Individual Explanations</p>
            {mockHotel.reasons.individuals.map((item, i) => (
              <p key={i}>• <span className="font-semibold">{item.user}:</span> {item.reason}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
