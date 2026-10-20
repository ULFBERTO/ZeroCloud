import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-sheet',
  standalone: true,
  imports: [],
  template: `
    @if (open()) {
      <div
        class="fixed inset-0 z-50 flex items-end"
        (click)="close.emit()"
      >
        <div class="absolute inset-0 bg-slate-950/60 backdrop-blur-sm transition-opacity" (click)="$event.stopPropagation()"></div>
        <div
          class="fixed right-0 top-0 h-full w-full sm:w-[420px] bg-slate-900 border-l border-slate-700 shadow-glass transform transition-transform duration-300 ease-out"
          [class.translate-x-full]="!open()"
          [class.translate-x-0]="open()"
          (click)="$event.stopPropagation()"
        >
          <div class="flex h-full flex-col">
            <div class="p-6 border-b border-slate-700/50 flex items-center justify-between">
              <h2 class="text-lg font-semibold text-slate-100">{{ title() }}</h2>
              <button (click)="close.emit()" class="-mr-1 p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors">✕</button>
            </div>
            <div class="flex-1 overflow-y-auto p-6">
              <ng-content />
            </div>
          </div>
        </div>
      </div>
    }
  `,
})
export class SheetComponent {
  readonly open = input<boolean>(false);
  readonly title = input<string>('');
  readonly close = output<void>();
}
