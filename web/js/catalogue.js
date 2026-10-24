// Common items with typical dimensions (metres), in the SPEC furniture format.
export const CATALOGUE = [
  {
    key: 'queen-bed', name: 'Queen bed', short: '163 x 213',
    assembled: { w: 1.63, d: 2.13, h: 1.05 }, flatpack: true, needs_assembly_space: 0.6,
    packages: [{ w: 2.12, d: 0.30, h: 0.16, count: 1 }, { w: 1.66, d: 0.62, h: 0.12, count: 1 }, { w: 1.62, d: 0.42, h: 0.10, count: 1 }],
  },
  {
    key: 'sofa-3', name: '3-seat sofa', short: '220 x 93',
    assembled: { w: 2.20, d: 0.93, h: 0.85 }, diagonal_depth: 0.76, flatpack: false,
  },
  {
    key: 'wardrobe', name: 'Wardrobe', short: '100 x 60 x 201',
    assembled: { w: 1.00, d: 0.60, h: 2.01 }, flatpack: true, needs_assembly_space: 0.7,
    packages: [{ w: 2.04, d: 0.61, h: 0.10, count: 2 }, { w: 1.00, d: 0.60, h: 0.08, count: 1 }],
  },
  {
    key: 'desk', name: 'Desk', short: '140 x 70',
    assembled: { w: 1.40, d: 0.70, h: 0.75 }, flatpack: true, needs_assembly_space: 0.5,
    packages: [{ w: 1.45, d: 0.75, h: 0.08, count: 1 }, { w: 0.75, d: 0.60, h: 0.12, count: 1 }],
  },
  {
    key: 'bookcase', name: 'Bookcase', short: '80 x 28 x 202',
    assembled: { w: 0.80, d: 0.28, h: 2.02 }, flatpack: true, needs_assembly_space: 0.5,
    packages: [{ w: 2.06, d: 0.32, h: 0.07, count: 1 }],
  },
  {
    key: 'fridge', name: 'Fridge', short: '70 x 73 x 178',
    assembled: { w: 0.70, d: 0.73, h: 1.78 }, flatpack: false,
  },
];

export const ITEM_COLOURS = ['#7d9ec4', '#c79c6d', '#8db48a', '#b48bbb', '#c8b35a', '#78b8b6', '#c48282', '#9b9cc4'];
