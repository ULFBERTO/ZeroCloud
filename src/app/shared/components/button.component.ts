import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-button',
  standalone: true,
  imports: [],
  template: `
    <button
      type="{{ type() }}"
      (click)="clicked.emit()"
      [disabled]="disabled()"
      class="inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed"
      [class]="variantClass()"
      [style.height.px]="size() === 'sm' ? 32 : size() === 'lg' ? 44 : 40"
      [style.min-width.px]="size() === 'sm' ? 80 : size() === 'lg' ? 120 : 100"
    >
      <span [class]="loading() ? 'opacity-50' : 'opacity-100'">{{ label() }}</span>
      @if (loading()) {
        <span class="ml-2 flex h-3 w-3"><svg class="animate-spin h-3 w-3 text-current" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg></span>
      }
    </button>
  `,
})
export class ButtonComponent {
  readonly label = input<string>('');
  readonly type = input<'button' | 'submit' | 'reset'>('button');
  readonly disabled = input<boolean>(false);
  readonly loading = input<boolean>(false);
  readonly clicked = output<void>();

  readonly variant = input<'primary' | 'secondary' | 'ghost' | 'danger' | 'outline'>('primary');
  readonly size = input<'sm' | 'md' | 'lg'>('md');

  variantClass(): string {
    switch (this.variant()) {
      case 'primary':
        return 'bg-indigo-600 text-slate-100 hover:bg-indigo-500 active:bg-indigo-700 shadow-md shadow-indigo-500/25';
      case 'secondary':
        return 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700';
      case 'ghost':
        return 'bg-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200';
      case 'danger':
        return 'bg-red-600 text-slate-100 hover:bg-red-500 active:bg-red-700';
      case 'outline':
        return 'border border-slate-600 bg-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200';
      default:
        return 'bg-indigo-600 text-slate-100';
    }
  }
}
